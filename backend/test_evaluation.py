import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import adapter_utils
import dataset_utils
import evaluation
import inference
import training
from server import app


def row(answer: str, evaluator_name: str | None = None, group_id: str | None = None) -> dict:
    value = {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": answer}]}
    if evaluator_name:
        value["evaluator"] = evaluator_name
    if group_id:
        value["group_id"] = group_id
    return value


def expert_row(group_id: str) -> dict:
    return {"task_profile": "expert", "evaluation_profile": "diagnostic", "group_id": group_id, "evaluator": "llm_judge", "messages": [{"role": "user", "content": "diagnose this"}, {"role": "assistant", "content": "reference answer"}]}


JUDGE_OUTPUT = """DIAGNOSIS: A=9 B=5
STEPS: A=8 B=4
SOLUTION: A=9 B=6
PLATFORM: A=NA B=NA
RELIABILITY: A=10 B=8
PREFERRED: A
REASON: A is more complete."""

STRUCTURED_JUDGE_OUTPUT = """CORRECTNESS: A=9 B=5
COMPLETENESS: A=8 B=5
FORMAT: A=9 B=4
PREFERRED: A
REASON: A matches the reference structure."""


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        root = Path(self.temporary.name)
        self.datasets = root / "datasets"
        self.adapters = root / "adapters"
        self.evaluations = root / "evaluations"
        self.datasets.mkdir(); self.adapters.mkdir()
        self.dataset_id = "a" * 32
        directory = self.datasets / self.dataset_id
        directory.mkdir()
        examples = [row("answer", "exact", "group"), row("A  B", "normalized_text", "group"), row('{"a":1,"b":2}', "json", "other")]
        for name in ("full", "train", "validation", "test"):
            (directory / f"{name}.jsonl").write_text("".join(json.dumps(value) + "\n" for value in examples), encoding="utf-8")
        (directory / "metadata.json").write_text('{"split":{}}', encoding="utf-8")
        adapter = self.adapters / "adapter"
        adapter.mkdir(); (adapter / "adapter_config.json").write_text("{}"); (adapter / "adapter_model.safetensors").write_text("x")
        self.patches = [patch.object(dataset_utils, "DATASETS_DIR", self.datasets), patch.object(adapter_utils, "ADAPTERS_DIR", self.adapters), patch.object(evaluation, "EVALUATIONS_DIR", self.evaluations)]
        for item in self.patches: item.start()
        evaluation._jobs.clear(); training._gpu_owner = None
        self.client = TestClient(app)

    def tearDown(self) -> None:
        training._gpu_owner = None; evaluation._jobs.clear()
        for item in reversed(self.patches): item.stop()
        self.temporary.cleanup()

    def _wait(self, evaluation_id: str) -> dict:
        for _ in range(100):
            value = self.client.get(f"/evaluations/{evaluation_id}").json()
            if value["status"] in {"completed", "failed"}: return value
            time.sleep(.01)
        self.fail("evaluation did not finish")

    def test_score_rules_and_fallback(self) -> None:
        self.assertTrue(evaluation.score("exact", " a ", "a"))
        self.assertFalse(evaluation.score("exact", "a b", "a  b"))
        self.assertTrue(evaluation.score("normalized_text", "Ａ　 b", "A b"))
        self.assertFalse(evaluation.score("normalized_text", "a!", "a"))
        self.assertTrue(evaluation.score("json", '{"a":1,"b":2}', '{"b":2,"a":1}'))
        self.assertFalse(evaluation.score("json", "{}", "not json"))
        with self.assertRaises(ValueError): evaluation.score("json", "not json", "{}")
        self.assertTrue(evaluation.score(None, "x", "x"))
        with self.assertRaises(ValueError): evaluation.score("fuzzy", "x", "x")

    def test_judge_parser_accepts_supported_forms_and_rejects_invalid_output(self) -> None:
        parsed = evaluation.parse_judge_output(JUDGE_OUTPUT.replace("A=9 B=5", "a = 9/10, b = 5.5/10"))
        self.assertEqual(parsed["criteria"]["diagnosis"], {"A": 9.0, "B": 5.5})
        self.assertIsNone(parsed["criteria"]["platform"]["A"])
        self.assertEqual(parsed["preferred"], "a")
        self.assertEqual(parsed["reason"], "A is more complete.")
        legacy_platform = evaluation.parse_judge_output(JUDGE_OUTPUT.replace("PLATFORM: A=NA B=NA", "PLATFORM: A|NA B|NA"))
        self.assertEqual(legacy_platform["criteria"]["platform"], {"A": None, "B": None})
        numeric_platform = evaluation.parse_judge_output(JUDGE_OUTPUT.replace("PLATFORM: A=NA B=NA", "PLATFORM: A=7 B=8"))
        self.assertEqual(numeric_platform["criteria"]["platform"], {"A": 7.0, "B": 8.0})
        for invalid in (
            JUDGE_OUTPUT.replace("A=9 B=5", "A=11 B=5"),
            JUDGE_OUTPUT.replace("PLATFORM: A=NA B=NA", "PLATFORM: A=NA B=8"),
            JUDGE_OUTPUT.replace("DIAGNOSIS: A=9 B=5", "DIAGNOSIS: A|8 B|7"),
            JUDGE_OUTPUT.replace("PLATFORM: A=NA B=NA\n", ""),
            "PREFERRED: A",
        ):
            with self.assertRaises(ValueError):
                evaluation.parse_judge_output(invalid)

    def test_expert_scoring_threshold_mapping_and_deterministic_order(self) -> None:
        output = """DIAGNOSIS: A=7 B=6.9
STEPS: A=7 B=6.9
SOLUTION: A=7 B=6.9
PLATFORM: A=NA B=NA
RELIABILITY: A=7 B=6.9
PREFERRED: A"""
        order = evaluation._judge_order("group", 2)
        self.assertEqual(order, evaluation._judge_order("group", 2))
        self.assertNotEqual(order, {"A": "base", "B": "base"})
        result = {}
        metrics = evaluation._empty_judge_metrics("diagnostic")
        evaluation._add_judge_result(result, order, output, metrics, True)
        self.assertTrue(result["judge_valid"])
        self.assertTrue(result["base_judge"]["passed"] or result["lora_judge"]["passed"])
        self.assertFalse(result["base_judge"]["passed"] and result["lora_judge"]["passed"])
        self.assertEqual(metrics["paired"]["total_examples"], 1)

    def test_expert_pipeline_is_blind_single_pass_and_records_diagnostics(self) -> None:
        examples = [expert_row("one"), expert_row("two"), expert_row("three")]
        directory = self.datasets / self.dataset_id
        for name in ("full", "train", "validation", "test"):
            (directory / f"{name}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in examples), encoding="utf-8")
        (directory / "metadata.json").write_text(json.dumps({"profile": {"task_profile": "expert", "evaluation_profile": "diagnostic", "evaluator": "llm_judge"}, "split": {}}), encoding="utf-8")
        calls = []
        answers = iter(["base"] * 3 + ["lora"] * 3 + [JUDGE_OUTPUT] * 3)
        def generate(messages, adapter_id, *_):
            calls.append((messages, adapter_id))
            return {"text": next(answers)}
        with patch.object(inference.inference_manager, "unload_for_training"), patch.object(inference.inference_manager, "generate_messages", side_effect=generate):
            response = self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"})
            job = self._wait(response.json()["evaluation_id"])
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["generation_config"]["judge_provider"], "local_base")
        self.assertEqual([adapter for _, adapter in calls], [None] * 3 + ["adapter"] * 3 + [None] * 3)
        self.assertTrue(all("Base" not in message[1]["content"] and "LoRA" not in message[1]["content"] and "adapter" not in message[1]["content"] for message, adapter in calls[6:]))
        results = self.client.get(f"/evaluations/{job['evaluation_id']}/results").json()
        self.assertEqual(len(results), 3)
        self.assertTrue(all(item["judge_valid"] for item in results))
        self.assertTrue(all(set(item["judge_order"].values()) == {"base", "lora"} for item in results))
        self.assertEqual(job["aggregate"]["expert"]["valid_judgements"], 3)
        self.assertEqual(job["aggregate"]["expert"]["per_criterion"]["platform"]["examples"], 0)

    def test_invalid_judge_output_is_excluded_from_quality_aggregate(self) -> None:
        examples = [expert_row("one"), expert_row("two"), expert_row("three")]
        directory = self.datasets / self.dataset_id
        for name in ("full", "train", "validation", "test"):
            (directory / f"{name}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in examples), encoding="utf-8")
        (directory / "metadata.json").write_text(json.dumps({"profile": {"task_profile": "expert", "evaluation_profile": "diagnostic"}, "split": {}}), encoding="utf-8")
        answers = iter(["base"] * 3 + ["lora"] * 3 + [JUDGE_OUTPUT, JUDGE_OUTPUT, "not a judge result"])
        with patch.object(inference.inference_manager, "unload_for_training"), patch.object(inference.inference_manager, "generate_messages", side_effect=lambda *_: {"text": next(answers)}):
            job = self._wait(self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"}).json()["evaluation_id"])
        self.assertEqual(job["status"], "completed")
        expert = job["aggregate"]["expert"]
        self.assertEqual(expert["total_examples"], 3)
        self.assertEqual(expert["valid_judgements"], 2)
        self.assertEqual(expert["invalid_judgements"], 1)
        self.assertEqual(expert["total_examples"], expert["valid_judgements"] + expert["invalid_judgements"])
        self.assertEqual(job["aggregate"]["total_examples"], 2)
        self.assertEqual(job["aggregate"]["both_wrong"], 0)
        self.assertEqual(job["aggregate"]["lora_improved"] + job["aggregate"]["lora_regressed"], 2)
        self.assertEqual(expert["base"]["pass_rate"], expert["base"]["pass_count"] / 2)
        self.assertEqual(expert["lora"]["pass_rate"], expert["lora"]["pass_count"] / 2)
        results = self.client.get(f"/evaluations/{job['evaluation_id']}/results").json()
        invalid = [item for item in results if not item["judge_valid"]]
        self.assertEqual(len(invalid), 1)
        self.assertIsNone(invalid[0]["base_judge"])
        self.assertIsNone(invalid[0]["lora_judge"])
        self.assertIsNone(invalid[0]["base_score"])
        self.assertIsNone(invalid[0]["lora_score"])

    def test_structured_json_metrics_handle_invalid_missing_extra_and_null(self) -> None:
        examples = []
        for index, target in enumerate(({"a": 1, "section": None}, {"a": 2, "section": "x"}, {"a": 3, "section": "y"})):
            examples.append(
                {
                    "task_profile": "structured_json",
                    "group_id": f"group-{index}",
                    "evaluator": "json",
                    "messages": [{"role": "user", "content": "input"}, {"role": "assistant", "content": json.dumps(target)}],
                }
            )
        directory = self.datasets / self.dataset_id
        for name in ("full", "train", "validation", "test"):
            (directory / f"{name}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in examples), encoding="utf-8")
        metadata = {
            "profile": {
                "task_profile": "structured_json",
                "target_schema": {"fields": ["a", "section"]},
            },
            "split": {},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        answers = iter([
            'JSON:\n{"a":1,"section":null}', "not json", '{"a":3,"section":"null"}',
            '{"a":1,"section":null,"extra":true}', '{"a":2}', '```json\n{"a":3,"section":"y"}\n```',
            STRUCTURED_JUDGE_OUTPUT, STRUCTURED_JUDGE_OUTPUT, "invalid judge response",
        ])
        calls = []
        def generate(messages, adapter_id, *_):
            calls.append((messages, adapter_id))
            return {"text": next(answers)}
        with patch.object(inference.inference_manager, "unload_for_training"), patch.object(
            inference.inference_manager, "generate_messages", side_effect=generate
        ):
            response = self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"})
            job = self._wait(response.json()["evaluation_id"])
        self.assertEqual(job["status"], "completed")
        structured = job["aggregate"]["structured_json"]
        self.assertEqual(structured["base"]["json_valid_count"], 2)
        self.assertEqual(structured["lora"]["json_valid_count"], 3)
        self.assertEqual(structured["base"]["raw_json_valid_count"], 1)
        self.assertEqual(structured["lora"]["raw_json_valid_count"], 2)
        self.assertEqual(structured["base"]["normalized_json_valid_count"], 2)
        self.assertEqual(structured["lora"]["normalized_json_valid_count"], 3)
        self.assertEqual(structured["base"]["normalization_required_count"], 1)
        self.assertEqual(structured["lora"]["normalization_required_count"], 1)
        self.assertEqual(structured["base"]["full_record_correct"], 1)
        self.assertEqual(structured["lora"]["full_record_correct"], 1)
        self.assertEqual(structured["base"]["field_accuracy"], 0.5)
        self.assertEqual(structured["lora"]["field_accuracy"], 5 / 6)
        self.assertEqual(structured["per_field"]["section"], {"total": 3, "base_correct": 1, "lora_correct": 2, "base_accuracy": 1 / 3, "lora_accuracy": 2 / 3})
        self.assertEqual(job["aggregate"]["judge"]["rubric"], "structured")
        judge = job["aggregate"]["judge"]
        self.assertEqual(judge["valid_judgements"], 2)
        self.assertEqual(judge["invalid_judgements"], 1)
        self.assertEqual(judge["paired"]["total_examples"], 2)
        results = self.client.get(f"/evaluations/{job['evaluation_id']}/results").json()
        invalid = next(item for item in results if not item["judge_valid"])
        self.assertFalse(invalid["base_correct"])
        self.assertTrue(invalid["lora_correct"])
        self.assertEqual(results[0]["base_normalization_steps"], ["strip_json_prefix"])
        self.assertFalse(results[0]["base_raw_json_valid"])
        self.assertTrue(results[0]["base_normalized_json_valid"])
        self.assertTrue(results[0]["base_normalization_applied"])
        self.assertEqual(results[2]["lora_normalization_steps"], ["strip_json_fence"])
        valid = [item for item in results if item["judge_valid"]]
        self.assertEqual(judge["base"]["pass_count"], sum(item["base_passed"] for item in valid))
        self.assertEqual(judge["lora"]["pass_count"], sum(item["lora_passed"] for item in valid))
        self.assertEqual(judge["paired"]["base_correct"], sum(item["base_passed"] for item in valid))
        self.assertEqual(judge["paired"]["lora_correct"], sum(item["lora_passed"] for item in valid))
        self.assertEqual([adapter for _, adapter in calls], [None] * 3 + ["adapter"] * 3 + [None] * 3)
        self.assertTrue(all("Base" not in messages[1]["content"] and "LoRA" not in messages[1]["content"] for messages, _ in calls[6:]))

    def test_structured_judge_paired_uses_judge_passes_not_deterministic_correctness(self) -> None:
        result = {"base_correct": True, "lora_correct": False}
        metrics = evaluation._empty_judge_metrics("structured")
        output = """CORRECTNESS: A=5 B=9
COMPLETENESS: A=5 B=9
FORMAT: A=5 B=9
PREFERRED: B"""
        evaluation._add_judge_result(result, {"A": "base", "B": "lora"}, output, metrics, False)
        self.assertTrue(result["base_correct"])
        self.assertFalse(result["lora_correct"])
        self.assertFalse(result["base_passed"])
        self.assertTrue(result["lora_passed"])
        self.assertEqual(metrics["paired"]["base_correct"], 0)
        self.assertEqual(metrics["paired"]["lora_correct"], 1)
        self.assertEqual(metrics["paired"]["lora_improved"], 1)

    def test_structured_json_normalization_is_safe_and_preserves_semantics(self) -> None:
        prose = evaluation._normalize_structured_output('Вот результат:\n{"a": 1}')
        outer = evaluation._normalize_structured_output('"{\\"a\\": 1}"')
        array = evaluation._normalize_structured_output('Ответ:\n[1, 2]')
        clean = evaluation._normalize_structured_output('{"a": 1}')
        ambiguous = evaluation._normalize_structured_output('{"a": 1}\n{"a": 2}')
        broken = evaluation._normalize_structured_output('{a: 1}')

        self.assertEqual(prose["payload"], {"a": 1})
        self.assertEqual(prose["normalization_steps"], ["extract_single_json_value"])
        self.assertFalse(prose["raw_json_valid"])
        self.assertTrue(prose["normalized_json_valid"])
        self.assertTrue(prose["normalization_applied"])
        self.assertEqual(outer["payload"], {"a": 1})
        self.assertEqual(outer["normalization_steps"], ["unwrap_outer_json_string"])
        self.assertEqual(array["payload"], [1, 2])
        self.assertTrue(array["normalized_json_valid"])
        self.assertTrue(clean["raw_json_valid"])
        self.assertTrue(clean["normalized_json_valid"])
        self.assertFalse(clean["normalization_applied"])
        self.assertEqual(clean["normalization_steps"], [])
        self.assertFalse(ambiguous["normalized_json_valid"])
        self.assertFalse(ambiguous["normalization_applied"])
        self.assertFalse(broken["normalized_json_valid"])
        self.assertFalse(broken["normalization_applied"])
        self.assertFalse(evaluation.score("json", '{"a": 2}', prose["text"]))

    def test_api_persists_order_metrics_and_isolates_input(self) -> None:
        calls = []
        phases = []
        lifecycle = []
        loads = []
        current = {"adapter": object()}
        answers = iter(["wrong", "A B", "not json", "answer", "A B", '{"b":2,"a":1}'])
        def generate(messages, adapter_id, *_):
            phases.append(next(iter(evaluation._jobs.values()))["phase"])
            lifecycle.append(f"generate:{adapter_id or 'base'}")
            if current["adapter"] != adapter_id:
                current["adapter"] = adapter_id
                loads.append(adapter_id)
            calls.append((messages, adapter_id)); return {"text": next(answers)}
        with patch.object(
            inference.inference_manager,
            "unload_for_training",
            side_effect=lambda: lifecycle.append("unload"),
        ), patch.object(inference.inference_manager, "generate_messages", side_effect=generate), patch.object(
            evaluation.logger,
            "info",
            side_effect=lambda message, *_: lifecycle.append(f"log:{message}"),
        ):
            response = self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter", "max_new_tokens": 9})
            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json()["phase"], "queued")
            self.assertEqual(response.json()["processed"], 0)
            job = self._wait(response.json()["evaluation_id"])
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["aggregate"]["total_examples"], 3)
        self.assertEqual(job["aggregate"]["base_accuracy"], 1 / 3)
        self.assertEqual(job["aggregate"]["lora_accuracy"], 1.0)
        self.assertEqual(job["aggregate"]["absolute_uplift"], job["aggregate"]["lora_accuracy"] - job["aggregate"]["base_accuracy"])
        self.assertEqual(job["aggregate"]["both_correct"] + job["aggregate"]["both_wrong"] + job["aggregate"]["lora_improved"] + job["aggregate"]["lora_regressed"], 3)
        self.assertEqual(job["aggregate"]["base_correct"], job["aggregate"]["both_correct"] + job["aggregate"]["lora_regressed"])
        self.assertEqual(job["aggregate"]["lora_correct"], job["aggregate"]["both_correct"] + job["aggregate"]["lora_improved"])
        results = self.client.get(f"/evaluations/{job['evaluation_id']}/results").json()
        self.assertEqual([item["index"] for item in results], [0, 1, 2])
        self.assertEqual([adapter for _, adapter in calls], [None] * 3 + ["adapter"] * 3)
        self.assertEqual(loads, [None, "adapter"])
        self.assertEqual(phases, ["loading_base", "evaluating_base", "evaluating_base", "loading_lora", "evaluating_lora", "evaluating_lora"])
        self.assertLess(lifecycle.index("unload", 1), lifecycle.index("generate:adapter"))
        self.assertLess(lifecycle.index("log:Evaluation %s: Releasing Base model"), lifecycle.index("unload", 1))
        self.assertLess(lifecycle.index("unload", 1), lifecycle.index("log:Evaluation %s: Base model released"))
        self.assertLess(lifecycle.index("log:Evaluation %s: Base model released"), lifecycle.index("log:Evaluation %s: Loading Base + LoRA model"))
        self.assertLess(lifecycle.index("log:Evaluation %s: Loading Base + LoRA model"), lifecycle.index("generate:adapter"))
        self.assertEqual(job["phase"], "completed")
        self.assertEqual(job["processed"], job["total"])
        self.assertTrue(all(len(messages) == 1 and messages[0]["content"] == "question" for messages, _ in calls))
        self.assertTrue((self.evaluations / job["evaluation_id"] / "metadata.json").is_file())
        self.assertTrue((self.evaluations / job["evaluation_id"] / "results.jsonl").is_file())
        evaluation._jobs.clear()
        self.assertEqual(self.client.get("/evaluations").json()[0]["evaluation_id"], job["evaluation_id"])
        self.assertEqual(self.client.get(f"/evaluations/{job['evaluation_id']}").json()["status"], "completed")
        self.assertEqual(len(self.client.get(f"/evaluations/{job['evaluation_id']}/results").json()), 3)

    def test_predefined_dataset_evaluates_only_test_split(self) -> None:
        directory = self.datasets / self.dataset_id
        def split_row(name: str) -> dict:
            return {"group_id": f"{name}-group", "evaluator": "exact", "messages": [{"role": "user", "content": f"{name} question"}, {"role": "assistant", "content": f"{name} answer"}]}
        train, validation, test = (split_row(name) for name in ("train", "validation", "test"))
        for name, value in (("train", train), ("validation", validation), ("test", test)):
            (directory / f"{name}.jsonl").write_text(json.dumps(value) + "\n", encoding="utf-8")
        (directory / "full.jsonl").write_text("".join(json.dumps(value) + "\n" for value in (train, validation, test)), encoding="utf-8")
        (directory / "metadata.json").write_text(json.dumps({"dataset_mode": "predefined", "split": {"seed": None}}), encoding="utf-8")
        messages = []
        with patch.object(inference.inference_manager, "unload_for_training"), patch.object(
            inference.inference_manager, "generate_messages", side_effect=lambda input_messages, *_: messages.append(input_messages) or {"text": "test answer"}
        ):
            job = self._wait(self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"}).json()["evaluation_id"])
        self.assertEqual(job["status"], "completed")
        self.assertEqual(messages, [[{"role": "user", "content": "test question"}]] * 2)
        self.assertNotIn("train question", json.dumps(messages))
        self.assertNotIn("validation question", json.dumps(messages))

    def test_missing_test_adapter_and_inference_failure_are_not_fake_metrics(self) -> None:
        self.assertEqual(self.client.post("/evaluations", json={"dataset_id": "b" * 32, "adapter_id": "adapter"}).status_code, 404)
        self.assertEqual(self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "missing"}).status_code, 404)
        invalid = self.adapters / "invalid"
        invalid.mkdir()
        self.assertEqual(self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "invalid"}).status_code, 400)
        (self.datasets / self.dataset_id / "test.jsonl").unlink()
        self.assertEqual(self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"}).status_code, 409)
        (self.datasets / self.dataset_id / "test.jsonl").write_text(json.dumps(row("answer")) + "\n")
        with patch.object(inference.inference_manager, "unload_for_training") as unload, patch.object(inference.inference_manager, "generate_messages", side_effect=RuntimeError("model failed")):
            response = self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"})
            job = self._wait(response.json()["evaluation_id"])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["phase"], "failed")
        self.assertIsNone(job["aggregate"])
        self.assertGreaterEqual(unload.call_count, 2)
        for _ in range(20):
            if training._gpu_owner is None:
                break
            time.sleep(.01)
        self.assertIsNone(training._gpu_owner)

    def test_base_answer_and_test_integrity_mismatches_fail_without_aggregate(self) -> None:
        original_progress = evaluation._progress

        def run_with_mutation(mutate):
            def progress(job, directory, phase, processed):
                original_progress(job, directory, phase, processed)
                if phase == "loading_lora":
                    mutate(directory)

            with patch.object(evaluation, "_progress", side_effect=progress), patch.object(
                inference.inference_manager, "unload_for_training"
            ), patch.object(inference.inference_manager, "generate_messages", return_value={"text": "answer"}):
                response = self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"})
                job = self._wait(response.json()["evaluation_id"])
            for _ in range(20):
                if training._gpu_owner is None:
                    break
                time.sleep(.01)
            return job

        short = run_with_mutation(
            lambda directory: (directory / "base_answers.jsonl").write_text(
                (directory / "base_answers.jsonl").read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8"
            )
        )
        self.assertEqual(short["status"], "failed")
        self.assertIsNone(short["aggregate"])
        self.assertIn("integrity mismatch", short["error"])

        def append_base_answer(directory):
            with (directory / "base_answers.jsonl").open("a", encoding="utf-8") as file:
                file.write('{"index":99,"base_answer":"x"}\n')

        long = run_with_mutation(append_base_answer)
        self.assertEqual(long["status"], "failed")
        self.assertIsNone(long["aggregate"])
        self.assertIn("integrity mismatch", long["error"])

    def test_test_hash_change_between_phases_fails_before_lora(self) -> None:
        original_progress = evaluation._progress

        def progress(job, directory, phase, processed):
            original_progress(job, directory, phase, processed)
            if phase == "loading_lora":
                with (self.datasets / self.dataset_id / "test.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row("answer")) + "\n")

        with patch.object(evaluation, "_progress", side_effect=progress), patch.object(
            inference.inference_manager, "unload_for_training"
        ), patch.object(inference.inference_manager, "generate_messages", return_value={"text": "answer"}) as generate:
            response = self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"})
            job = self._wait(response.json()["evaluation_id"])
        self.assertEqual(job["status"], "failed")
        self.assertIsNone(job["aggregate"])
        self.assertIn("test hash integrity mismatch", job["error"])
        self.assertEqual([call.args[1] for call in generate.call_args_list], [None] * 3)

    def test_reconciliation_interrupts_only_active_persisted_jobs(self) -> None:
        self.evaluations.mkdir()
        jobs = {
            "b" * 32: {"status": "running", "phase": "evaluating_base"},
            "c" * 32: {"status": "queued", "phase": "queued"},
            "d" * 32: {"status": "completed", "phase": "finalizing"},
            "e" * 32: {"status": "failed", "phase": "loading_lora"},
            "f" * 32: {"status": "interrupted", "phase": "evaluating_lora"},
        }
        for evaluation_id, job in jobs.items():
            directory = self.evaluations / evaluation_id
            directory.mkdir()
            (directory / "metadata.json").write_text(json.dumps(job), encoding="utf-8")

        self.assertEqual(evaluation.reconcile_evaluations(), 2)
        for evaluation_id in ("b" * 32, "c" * 32):
            job = evaluation.get_evaluation(evaluation_id)
            self.assertEqual(job["status"], "interrupted")
            self.assertEqual(job["phase"], "interrupted")
            self.assertEqual(job["error"], "Backend restarted while evaluation was running")
        self.assertEqual(evaluation.get_evaluation("d" * 32)["status"], "completed")
        self.assertEqual(evaluation.get_evaluation("d" * 32)["phase"], "completed")
        self.assertEqual(evaluation.get_evaluation("e" * 32)["status"], "failed")
        self.assertEqual(evaluation.get_evaluation("e" * 32)["phase"], "failed")
        self.assertEqual(evaluation.get_evaluation("f" * 32)["status"], "interrupted")
        self.assertEqual(evaluation.get_evaluation("f" * 32)["phase"], "interrupted")

    def test_partial_lora_results_are_not_published(self) -> None:
        calls = 0

        def generate(*_):
            nonlocal calls
            calls += 1
            if calls > 3:
                raise RuntimeError("LoRA failed")
            return {"text": "answer"}

        with patch.object(inference.inference_manager, "unload_for_training"), patch.object(
            inference.inference_manager, "generate_messages", side_effect=generate
        ):
            response = self.client.post("/evaluations", json={"dataset_id": self.dataset_id, "adapter_id": "adapter"})
            job = self._wait(response.json()["evaluation_id"])

        self.assertEqual(job["status"], "failed")
        self.assertEqual(self.client.get(f"/evaluations/{job['evaluation_id']}/results").json(), [])
        self.assertFalse((self.evaluations / job["evaluation_id"] / "results.jsonl").exists())
