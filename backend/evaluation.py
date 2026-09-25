"""Persistent deterministic Base-versus-LoRA evaluation jobs."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import threading
import unicodedata
from itertools import zip_longest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapter_utils import adapter_path, adapter_validation
from dataset_utils import content_hash, dataset_path, dataset_profile, split_path, validate_jsonl
from training import GpuBusyError, release_evaluation, reserve_evaluation


logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
EVALUATIONS_DIR = BACKEND_DIR / "data" / "evaluations"
EVALUATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
ACTIVE_STATUSES = {"queued", "running"}
ACTIVE_PHASES = {
    "queued",
    "loading_base",
    "evaluating_base",
    "releasing_base",
    "loading_lora",
    "evaluating_lora",
    "releasing_lora",
    "loading_judge",
    "evaluating_judge",
    "finalizing",
}
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()
JUDGE_PASS_THRESHOLD = 0.70
JUDGE_RUBRICS = {
    "structured": {
        "weights": {"correctness": 0.60, "completeness": 0.25, "format": 0.15},
        "instruction": "Assess factual correctness against the reference, completeness of required values, and conformance to the requested structured output. Do not prefer wording, length, or style.",
    },
    "diagnostic": {
        "weights": {"diagnosis": 0.30, "steps": 0.20, "solution": 0.25, "platform": 0.10, "reliability": 0.15},
        "instruction": "Assess diagnosis, steps, solution, platform applicability, and reliability. Do not prefer wording, length, or style.",
    },
}
LOCAL_BASE_JUDGE_PROVIDER = "local_base"


def start_evaluation(dataset_id: str, adapter_id: str, max_new_tokens: int) -> dict[str, Any]:
    dataset_path(dataset_id)
    test_path = split_path(dataset_id, "test")
    validation = validate_jsonl(test_path)
    if not validation["valid"]:
        raise ValueError("Canonical test dataset validation failed")
    adapter = adapter_path(adapter_id)
    valid, error = adapter_validation(adapter)
    if not valid:
        raise ValueError(f"Invalid PEFT adapter files: {error}")
    profile = dataset_profile(dataset_id)
    structured_fields = (
        profile.get("target_schema", {}).get("fields")
        if profile and profile.get("task_profile") == "structured_json"
        else None
    )
    task_profile = profile.get("task_profile") if profile else None
    evaluation_profile = profile.get("evaluation_profile") if profile else None
    judge_rubric = "structured" if task_profile == "structured_json" else "diagnostic" if task_profile == "expert" else None

    with _lock:
        if any(job["status"] in ACTIVE_STATUSES for job in _jobs.values()):
            raise RuntimeError("An evaluation job is already active")
        reserve_evaluation()
        evaluation_id = uuid4().hex
        directory = _directory(evaluation_id)
        try:
            directory.mkdir(parents=True)
            job = {
                "evaluation_id": evaluation_id,
                "dataset_id": dataset_id,
                "adapter_id": adapter_id,
                "created_at": _timestamp(),
                "status": "queued",
                "phase": "queued",
                "processed": 0,
                "total": validation["n_examples"],
                "test_count": validation["n_examples"],
                "test_hash": content_hash(test_path),
                "generation_config": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "do_sample": False,
                    **({"judge_provider": LOCAL_BASE_JUDGE_PROVIDER, "judge_max_new_tokens": 256, "judge_temperature": 0, "judge_passes": 1, "judge_model": "base"} if judge_rubric else {}),
                },
                "aggregate": None,
                "structured_fields": structured_fields,
                "task_profile": task_profile,
                "evaluation_profile": evaluation_profile,
                "judge_rubric": judge_rubric,
                "error": None,
            }
            _write_metadata(directory, job)
            _jobs[evaluation_id] = job
            threading.Thread(target=_run, args=(evaluation_id, test_path), daemon=True).start()
            return job.copy()
        except Exception:
            release_evaluation()
            raise


def list_evaluations() -> list[dict[str, Any]]:
    with _lock:
        EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
        return [_read_metadata(path) for path in sorted(EVALUATIONS_DIR.iterdir()) if path.is_dir() and not path.is_symlink() and EVALUATION_ID_PATTERN.fullmatch(path.name) and _read_metadata(path)]


def get_evaluation(evaluation_id: str) -> dict[str, Any] | None:
    with _lock:
        directory = _directory(evaluation_id)
        if not directory.is_dir():
            return None
        return _read_metadata(directory)


def evaluation_results(evaluation_id: str) -> list[dict[str, Any]]:
    with _lock:
        directory = _directory(evaluation_id)
        path = directory / "results.jsonl"
        if not directory.is_dir():
            raise FileNotFoundError(evaluation_id)
        metadata = _read_metadata(directory)
        if not metadata or metadata.get("status") != "completed":
            return []
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file]


def reconcile_evaluations() -> int:
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    interrupted = 0
    for directory in EVALUATIONS_DIR.iterdir():
        if not directory.is_dir() or directory.is_symlink() or not EVALUATION_ID_PATTERN.fullmatch(directory.name):
            continue
        job = _read_metadata(directory)
        if not job:
            continue
        if job.get("status") in {"completed", "failed", "interrupted"}:
            if job.get("phase") != job["status"]:
                job["phase"] = job["status"]
                _write_metadata(directory, job)
            continue
        if job.get("status") in ACTIVE_STATUSES or job.get("phase") in ACTIVE_PHASES:
            job["status"] = "interrupted"
            job["phase"] = "interrupted"
            job["error"] = "Backend restarted while evaluation was running"
            _write_metadata(directory, job)
            interrupted += 1
    return interrupted


def score(evaluator: str | None, expected: str, actual: str) -> bool:
    evaluator = evaluator or "exact"
    if evaluator == "exact":
        return expected.strip() == actual.strip()
    if evaluator == "normalized_text":
        return _normalize(expected) == _normalize(actual)
    if evaluator == "json":
        try:
            expected_value = json.loads(expected)
        except json.JSONDecodeError:
            raise ValueError("Expected answer is not valid JSON")
        try:
            return expected_value == json.loads(actual)
        except json.JSONDecodeError:
            return False
    raise ValueError(f"Unsupported evaluator: {evaluator}")


def _run(evaluation_id: str, test_path: Path) -> None:
    directory = _directory(evaluation_id)
    with _lock:
        job = _jobs[evaluation_id]
        job["status"] = "running"
        _write_metadata(directory, job)
    try:
        from inference import inference_manager

        inference_manager.unload_for_training()
        config = job["generation_config"]
        base_path = directory / "base_answers.jsonl"
        base_temporary = directory / ".base_answers.tmp"
        _progress(job, directory, "loading_base", 0)
        base_count = 0
        with test_path.open(encoding="utf-8") as source, base_temporary.open("w", encoding="utf-8") as answers:
            for index, line in enumerate(source):
                example = _example(line, index)
                base = inference_manager.generate_messages(example["input_messages"], None, config["max_new_tokens"], 0)["text"]
                answers.write(json.dumps({"index": index, "base_answer": base}, ensure_ascii=False) + "\n")
                answers.flush()
                base_count = index + 1
                _progress(job, directory, "evaluating_base", index + 1)
        if base_count != job["test_count"]:
            raise RuntimeError("Base answer and canonical test integrity mismatch")
        os.replace(base_temporary, base_path)
        logger.info("Evaluation %s: Base evaluation completed", evaluation_id)

        aggregate = _empty_aggregate()
        structured = _empty_structured_metrics(job["structured_fields"])
        judge_metrics = _empty_judge_metrics(job["judge_rubric"]) if job.get("judge_rubric") else None
        legacy_expert = job.get("task_profile") == "expert"
        _progress(job, directory, "releasing_base", job["total"])
        logger.info("Evaluation %s: Releasing Base model", evaluation_id)
        inference_manager.unload_for_training()
        logger.info("Evaluation %s: Base model released", evaluation_id)
        logger.info("Evaluation %s: Loading Base + LoRA model", evaluation_id)
        _progress(job, directory, "loading_lora", 0)
        if content_hash(test_path) != job["test_hash"]:
            raise RuntimeError("Canonical test hash integrity mismatch before LoRA phase")
        results_path = directory / "results.jsonl"
        results_temporary = directory / ".results.tmp"
        with (
            test_path.open(encoding="utf-8") as source,
            base_path.open(encoding="utf-8") as answers,
            results_temporary.open("w", encoding="utf-8") as results,
        ):
            for index, (line, base_line) in enumerate(zip_longest(source, answers)):
                if line is None or base_line is None:
                    raise RuntimeError("Base answer and canonical test integrity mismatch")
                example = _example(line, index)
                base_record = json.loads(base_line)
                if base_record.get("index") != index:
                    raise RuntimeError("Base answer order does not match canonical test order")
                lora = inference_manager.generate_messages(example["input_messages"], job["adapter_id"], config["max_new_tokens"], 0)["text"]
                base_text = str(base_record["base_answer"])
                lora_text = str(lora)
                base_normalization = _normalize_structured_output(base_text) if structured is not None else None
                lora_normalization = _normalize_structured_output(lora_text) if structured is not None else None
                scored_base = base_normalization["text"] if base_normalization else base_text
                scored_lora = lora_normalization["text"] if lora_normalization else lora_text
                base_correct = score(example["evaluator"], example["expected"], scored_base) if not legacy_expert else False
                lora_correct = score(example["evaluator"], example["expected"], scored_lora) if not legacy_expert else False
                if structured is not None:
                    _add_structured_metrics(
                        structured,
                        example["expected"],
                        base_normalization,
                        lora_normalization,
                    )
                result = {"index": index, "evaluator": example["evaluator"], "input_messages": example["input_messages"], "expected": example["expected"], "base_answer": base_record["base_answer"], "lora_answer": lora, "base_correct": base_correct, "lora_correct": lora_correct}
                if base_normalization is not None:
                    for name, normalization in (("base", base_normalization), ("lora", lora_normalization)):
                        result[f"{name}_raw_json_valid"] = normalization["raw_json_valid"]
                        result[f"{name}_normalized_json_valid"] = normalization["normalized_json_valid"]
                        result[f"{name}_normalization_applied"] = normalization["normalization_applied"]
                        result[f"{name}_normalization_steps"] = normalization["normalization_steps"]
                if example["group_id"] is not None:
                    result["group_id"] = example["group_id"]
                results.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.flush()
                _add(aggregate, base_correct, lora_correct)
                _progress(job, directory, "evaluating_lora", index + 1)
        if aggregate["total_examples"] != job["test_count"]:
            raise RuntimeError("Paired result count integrity mismatch")
        if judge_metrics is not None:
            _progress(job, directory, "releasing_lora", job["total"])
            inference_manager.unload_for_training()
            _progress(job, directory, "loading_judge", 0)
            judged_temporary = directory / ".judged_results.tmp"
            with results_temporary.open(encoding="utf-8") as source, judged_temporary.open("w", encoding="utf-8") as output:
                for index, line in enumerate(source):
                    result = json.loads(line)
                    judge_order = _judge_order(result.get("group_id"), index)
                    answer_a = result["base_answer"] if judge_order["A"] == "base" else result["lora_answer"]
                    answer_b = result["lora_answer"] if judge_order["B"] == "lora" else result["base_answer"]
                    judge_text = _judge_answer(
                        config["judge_provider"],
                        inference_manager,
                        _judge_messages(result["input_messages"], result["expected"], answer_a, answer_b, job["judge_rubric"]),
                        config,
                    )
                    _add_judge_result(result, judge_order, str(judge_text), judge_metrics, legacy_expert)
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    _progress(job, directory, "evaluating_judge", index + 1)
            os.replace(judged_temporary, results_temporary)
            if legacy_expert:
                aggregate = judge_metrics["paired"]
        _check(aggregate)
        os.replace(results_temporary, results_path)
        with _lock:
            _progress(job, directory, "finalizing", job["total"])
            job["aggregate"] = _finish(aggregate)
            if structured is not None:
                job["aggregate"]["structured_json"] = _finish_structured_metrics(structured)
            if judge_metrics is not None:
                finished_judge = _finish_judge_metrics(judge_metrics, config["judge_provider"])
                job["aggregate"]["judge"] = finished_judge
                if legacy_expert:
                    job["aggregate"]["expert"] = {**finished_judge, "evaluation_profile": "diagnostic"}
            job["status"] = "completed"
            job["phase"] = "completed"
            _write_metadata(directory, job)
    except Exception as error:
        with _lock:
            job = _jobs.get(evaluation_id)
            if job:
                job["status"] = "failed"
                job["phase"] = "failed"
                job["error"] = str(error)
                _write_metadata(directory, job)
    finally:
        try:
            from inference import inference_manager

            inference_manager.unload_for_training()
        except Exception:
            pass
        finally:
            try:
                for path in (directory / ".base_answers.tmp", directory / ".results.tmp", directory / ".judged_results.tmp"):
                    path.unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                release_evaluation()


def _empty_aggregate() -> dict[str, int]:
    return {name: 0 for name in ("total_examples", "base_correct", "lora_correct", "both_correct", "both_wrong", "lora_improved", "lora_regressed")}


def _add(metrics: dict[str, int], base: bool, lora: bool) -> None:
    metrics["total_examples"] += 1
    metrics["base_correct"] += base
    metrics["lora_correct"] += lora
    if base and lora:
        metrics["both_correct"] += 1
    elif not base and not lora:
        metrics["both_wrong"] += 1
    elif lora:
        metrics["lora_improved"] += 1
    else:
        metrics["lora_regressed"] += 1


def _finish(metrics: dict[str, int]) -> dict[str, Any]:
    total = metrics["total_examples"]
    base_accuracy = metrics["base_correct"] / total if total else None
    lora_accuracy = metrics["lora_correct"] / total if total else None
    return {**metrics, "base_accuracy": base_accuracy, "lora_accuracy": lora_accuracy, "absolute_uplift": lora_accuracy - base_accuracy if total else None}


def _check(metrics: dict[str, int]) -> None:
    if metrics["both_correct"] + metrics["both_wrong"] + metrics["lora_improved"] + metrics["lora_regressed"] != metrics["total_examples"]:
        raise RuntimeError("Invalid paired evaluation metrics")
    if metrics["base_correct"] != metrics["both_correct"] + metrics["lora_regressed"]:
        raise RuntimeError("Invalid base evaluation metrics")
    if metrics["lora_correct"] != metrics["both_correct"] + metrics["lora_improved"]:
        raise RuntimeError("Invalid LoRA evaluation metrics")


def _empty_structured_metrics(fields: list[str] | None) -> dict[str, Any] | None:
    if fields is None:
        return None
    return {
        "total_examples": 0,
        "base": {"json_valid_count": 0, "raw_json_valid_count": 0, "normalized_json_valid_count": 0, "normalization_required_count": 0, "full_record_correct": 0, "field_correct": 0, "field_total": 0},
        "lora": {"json_valid_count": 0, "raw_json_valid_count": 0, "normalized_json_valid_count": 0, "normalization_required_count": 0, "full_record_correct": 0, "field_correct": 0, "field_total": 0},
        "per_field": {field: {"total": 0, "base_correct": 0, "lora_correct": 0} for field in fields},
    }


def _add_structured_metrics(
    metrics: dict[str, Any], expected_text: str, base_normalization: dict[str, Any], lora_normalization: dict[str, Any]
) -> None:
    expected = _json_object(expected_text)
    if expected is None:
        raise ValueError("Structured JSON expected target is not a JSON object")
    fields = metrics["per_field"]
    if sorted(expected) != list(fields):
        raise ValueError("Structured JSON expected target does not match dataset schema")
    metrics["total_examples"] += 1
    for field_summary in fields.values():
        field_summary["total"] += 1
    for name, normalization in (("base", base_normalization), ("lora", lora_normalization)):
        actual = normalization["payload"] if isinstance(normalization["payload"], dict) else None
        summary = metrics[name]
        summary["raw_json_valid_count"] += normalization["raw_json_valid"]
        summary["normalized_json_valid_count"] += normalization["normalized_json_valid"]
        summary["normalization_required_count"] += normalization["normalization_applied"]
        if actual is not None:
            summary["json_valid_count"] += 1
            if actual == expected:
                summary["full_record_correct"] += 1
        for field, field_summary in fields.items():
            summary["field_total"] += 1
            if actual is not None and field in actual and actual[field] == expected[field]:
                summary["field_correct"] += 1
                field_summary[f"{name}_correct"] += 1


def _finish_structured_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    total = metrics["total_examples"]
    result = {"total_examples": total}
    for name in ("base", "lora"):
        summary = metrics[name]
        result[name] = {
            **summary,
            "json_valid_rate": summary["json_valid_count"] / total,
            "raw_json_valid_rate": summary["raw_json_valid_count"] / total,
            "normalized_json_valid_rate": summary["normalized_json_valid_count"] / total,
            "normalization_required_rate": summary["normalization_required_count"] / total,
            "full_record_accuracy": summary["full_record_correct"] / total,
            "field_accuracy": summary["field_correct"] / summary["field_total"],
        }
    result["per_field"] = {
        field: {
            **summary,
            "base_accuracy": summary["base_correct"] / summary["total"],
            "lora_accuracy": summary["lora_correct"] / summary["total"],
        }
        for field, summary in metrics["per_field"].items()
    }
    return result


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_value(value: str) -> dict[str, Any] | list[Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _single_embedded_json_value(value: str) -> tuple[str, dict[str, Any] | list[Any]] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character not in "[{":
            continue
        try:
            parsed, end = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            if not value[index + end :].strip():
                return value[index : index + end], parsed
            return None
    return None


def _normalize_structured_output(value: str) -> dict[str, Any]:
    stripped = value.strip()
    raw = _json_value(value)
    if raw is not None:
        return {
            "text": value,
            "payload": raw,
            "raw_json_valid": True,
            "normalized_json_valid": True,
            "normalization_applied": False,
            "normalization_steps": [],
        }

    candidate: str | None = None
    step: str | None = None
    if stripped.lower().startswith("json:"):
        candidate = stripped[5:].strip()
        step = "strip_json_prefix"
    elif stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"}:
            candidate = "\n".join(lines[1:-1]).strip()
            step = "strip_json_fence"
    else:
        try:
            outer = json.loads(stripped)
        except json.JSONDecodeError:
            outer = None
        if isinstance(outer, str):
            candidate = outer
            step = "unwrap_outer_json_string"

    payload = _json_value(candidate) if candidate is not None else None
    if payload is None and candidate is None:
        embedded = _single_embedded_json_value(stripped)
        if embedded is not None:
            candidate, payload = embedded
            step = "extract_single_json_value"
    if payload is not None:
        return {
            "text": candidate,
            "payload": payload,
            "raw_json_valid": False,
            "normalized_json_valid": True,
            "normalization_applied": True,
            "normalization_steps": [step],
        }
    return {
        "text": value,
        "payload": None,
        "raw_json_valid": False,
        "normalized_json_valid": False,
        "normalization_applied": False,
        "normalization_steps": [],
    }


def _judge_order(group_id: str | None, index: int) -> dict[str, str]:
    token = hashlib.sha256(f"{group_id or ''}:{index}".encode()).digest()[0]
    return {"A": "base", "B": "lora"} if token % 2 == 0 else {"A": "lora", "B": "base"}


def _judge_messages(
    input_messages: list[dict[str, str]], reference: str, answer_a: str, answer_b: str, rubric: str
) -> list[dict[str, str]]:
    specification = JUDGE_RUBRICS[rubric]
    criteria = tuple(specification["weights"])
    score_lines = "\n".join(f"{name.upper()}: A=<score> B=<score>" for name in criteria if name != "platform")
    platform_line = (
        "For PLATFORM, use PLATFORM: A=<score> B=<score> when it applies to both answers. "
        "If it applies to neither answer, use exactly PLATFORM: A=NA B=NA.\n"
        if "platform" in criteria
        else ""
    )
    instruction = (
        "You are an independent technical evaluator. Compare two anonymous answers with a task and reference answer. "
        f"{specification['instruction']} "
        "Return exactly these score lines using 0..10 (decimals and /10 are allowed):\n"
        f"{score_lines}\n{platform_line}"
        "Choose exactly one: PREFERRED: A, PREFERRED: B, or PREFERRED: TIE."
    )
    task = json.dumps(input_messages, ensure_ascii=False)
    content = f"TASK:\n{task}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER A:\n{answer_a}\n\nANSWER B:\n{answer_b}"
    return [{"role": "system", "content": instruction}, {"role": "user", "content": content}]


def _judge_answer(
    provider: str, inference_manager: Any, messages: list[dict[str, str]], config: dict[str, Any]
) -> str:
    if provider != LOCAL_BASE_JUDGE_PROVIDER:
        raise ValueError(f"Unsupported judge provider: {provider}")
    return str(
        inference_manager.generate_messages(
            messages, None, config["judge_max_new_tokens"], config["judge_temperature"]
        )["text"]
    )


def parse_judge_output(text: str, rubric: str = "diagnostic") -> dict[str, Any]:
    specification = JUDGE_RUBRICS[rubric]
    scores: dict[str, dict[str, float | None]] = {}
    for criterion in specification["weights"]:
        match = re.search(
            rf"(?im)^\s*{criterion}\s*:\s*A\s*=\s*(NA|\d+(?:\.\d+)?(?:\s*/\s*10)?)\s*[,;]?\s*B\s*=\s*(NA|\d+(?:\.\d+)?(?:\s*/\s*10)?)",
            text,
        )
        if criterion == "platform" and not match:
            match = re.search(r"(?im)^\s*platform\s*:\s*A\s*\|\s*(NA)\s*[,;]?\s*B\s*\|\s*(NA)\s*$", text)
        if not match:
            raise ValueError(f"Judge output is missing {criterion.upper()} scores")
        left, right = (_judge_score(value) for value in match.groups())
        if criterion == "platform":
            if (left is None) != (right is None):
                raise ValueError("Judge PLATFORM must be NA for both answers or neither")
        elif left is None or right is None:
            raise ValueError(f"Judge {criterion.upper()} cannot be NA")
        scores[criterion] = {"A": left, "B": right}
    preferred = re.search(r"(?im)^\s*PREFERRED\s*:\s*(A|B|TIE)\s*$", text)
    if not preferred:
        raise ValueError("Judge output is missing PREFERRED")
    reason = re.search(r"(?im)^\s*REASON\s*:\s*(.+)$", text)
    return {"criteria": scores, "preferred": preferred.group(1).lower(), "reason": reason.group(1).strip() if reason else None}


def _judge_score(value: str) -> float | None:
    if value.upper() == "NA":
        return None
    score = float(value.split("/")[0].strip())
    if not 0 <= score <= 10:
        raise ValueError("Judge score must be between 0 and 10")
    return score


def _overall(criteria: dict[str, dict[str, float | None]], answer: str, weights: dict[str, float]) -> float:
    applicable = {name: weight for name, weight in weights.items() if criteria[name][answer] is not None}
    return sum(criteria[name][answer] * weight for name, weight in applicable.items()) / sum(applicable.values()) / 10


def _add_judge_result(
    result: dict[str, Any], order: dict[str, str], judge_text: str, metrics: dict[str, Any], replace_correctness: bool
) -> None:
    result["judge_order"] = order
    result["judge_raw"] = judge_text
    metrics["total_examples"] += 1
    try:
        parsed = parse_judge_output(judge_text, metrics["rubric"])
        criteria = parsed["criteria"]
        base_key = "A" if order["A"] == "base" else "B"
        lora_key = "A" if order["A"] == "lora" else "B"
        base_score, lora_score = _overall(criteria, base_key, metrics["weights"]), _overall(criteria, lora_key, metrics["weights"])
        base_judge = {name: (values[base_key] / 10 if values[base_key] is not None else None) for name, values in criteria.items()}
        lora_judge = {name: (values[lora_key] / 10 if values[lora_key] is not None else None) for name, values in criteria.items()}
        base_judge.update({"overall": base_score, "passed": base_score >= JUDGE_PASS_THRESHOLD})
        lora_judge.update({"overall": lora_score, "passed": lora_score >= JUDGE_PASS_THRESHOLD})
        result.update({
            "judge_valid": True,
            "judge_error": None,
            "base_judge": base_judge,
            "lora_judge": lora_judge,
            "base_score": base_score,
            "lora_score": lora_score,
            "base_passed": base_score >= JUDGE_PASS_THRESHOLD,
            "lora_passed": lora_score >= JUDGE_PASS_THRESHOLD,
        })
        if replace_correctness:
            result["base_correct"] = result["base_passed"]
            result["lora_correct"] = result["lora_passed"]
        preferred = order[parsed["preferred"].upper()] if parsed["preferred"] != "tie" else "tie"
        calculated = "tie" if abs(base_score - lora_score) < 0.01 else ("base" if base_score > lora_score else "lora")
        result.update({"judge_preferred": preferred, "calculated_preferred": calculated, "preference_agreement": preferred == calculated, "judge_reason": parsed["reason"]})
        _add(metrics["paired"], result["base_passed"], result["lora_passed"])
        metrics["valid_judge_count"] += 1
        metrics["base_score_sum"] += base_score
        metrics["lora_score_sum"] += lora_score
        for name in metrics["weights"]:
            if base_judge[name] is not None:
                metrics["criteria"][name]["examples"] += 1
                metrics["criteria"][name]["base_sum"] += base_judge[name]
                metrics["criteria"][name]["lora_sum"] += lora_judge[name]
        metrics["judge_preference"][preferred] += 1
        metrics["calculated_preference"][calculated] += 1
        metrics["preference_agreement_count"] += result["preference_agreement"]
    except ValueError as error:
        result.update({"judge_valid": False, "judge_error": str(error), "base_judge": None, "lora_judge": None, "base_score": None, "lora_score": None, "base_passed": None, "lora_passed": None, "judge_reason": None})
        if replace_correctness:
            result.update({"base_correct": False, "lora_correct": False})
        metrics["invalid_judge_count"] += 1


def _empty_judge_metrics(rubric: str) -> dict[str, Any]:
    weights = JUDGE_RUBRICS[rubric]["weights"]
    return {"rubric": rubric, "weights": weights, "total_examples": 0, "paired": _empty_aggregate(), "valid_judge_count": 0, "invalid_judge_count": 0, "base_score_sum": 0.0, "lora_score_sum": 0.0, "judge_preference": {"base": 0, "lora": 0, "tie": 0}, "calculated_preference": {"base": 0, "lora": 0, "tie": 0}, "preference_agreement_count": 0, "criteria": {name: {"examples": 0, "base_sum": 0.0, "lora_sum": 0.0} for name in weights}}


def _finish_judge_metrics(metrics: dict[str, Any], provider: str) -> dict[str, Any]:
    valid = metrics["valid_judge_count"]
    paired = metrics["paired"]
    return {
        "rubric": metrics["rubric"],
        "judge": {"provider": provider, "type": "base_model", "passes": 1, "temperature": 0},
        "total_examples": metrics["total_examples"],
        "valid_judgements": valid,
        "invalid_judgements": metrics["invalid_judge_count"],
        "pass_threshold": JUDGE_PASS_THRESHOLD,
        "base": {"mean_score": metrics["base_score_sum"] / valid if valid else None, "pass_count": paired["base_correct"], "pass_rate": paired["base_correct"] / valid if valid else None},
        "lora": {"mean_score": metrics["lora_score_sum"] / valid if valid else None, "pass_count": paired["lora_correct"], "pass_rate": paired["lora_correct"] / valid if valid else None},
        "mean_uplift": (metrics["lora_score_sum"] - metrics["base_score_sum"]) / valid if valid else None,
        "paired": _finish(paired),
        "judge_preference": metrics["judge_preference"],
        "calculated_preference": metrics["calculated_preference"],
        "preference_agreement_rate": metrics["preference_agreement_count"] / valid if valid else None,
        "per_criterion": {name: {"examples": values["examples"], "base_mean_score": values["base_sum"] / values["examples"] if values["examples"] else None, "lora_mean_score": values["lora_sum"] / values["examples"] if values["examples"] else None} for name, values in metrics["criteria"].items()},
    }


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def _example(line: str, index: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
        messages = value["messages"]
        if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError
        if not isinstance(messages[-1].get("content"), str):
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid test example at line {index + 1}") from error
    return {"input_messages": messages[:-1], "expected": messages[-1]["content"], "evaluator": value.get("evaluator", "exact"), "group_id": value.get("group_id")}


def _progress(job: dict[str, Any], directory: Path, phase: str, processed: int) -> None:
    with _lock:
        job["phase"] = phase
        job["processed"] = processed
        _write_metadata(directory, job)


def _directory(evaluation_id: str) -> Path:
    if not EVALUATION_ID_PATTERN.fullmatch(evaluation_id):
        raise ValueError("Invalid evaluation id")
    path = (EVALUATIONS_DIR / evaluation_id).resolve()
    try:
        path.relative_to(EVALUATIONS_DIR.resolve())
    except ValueError as error:
        raise ValueError("Evaluation path must stay inside the evaluations directory") from error
    return path


def _write_metadata(directory: Path, data: dict[str, Any]) -> None:
    temporary = directory / ".metadata.tmp"
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, directory / "metadata.json")


def _read_metadata(directory: Path) -> dict[str, Any] | None:
    try:
        value = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
