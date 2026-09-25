"""Validation and safe local storage for JSONL datasets."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

DATASETS_DIR = Path(__file__).resolve().parent / "data" / "datasets"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
DATASET_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_EVALUATORS = {"exact", "normalized_text", "json", "llm_judge"}
STRUCTURED_JSON_PROFILE = "structured_json"
EXPERT_PROFILE = "expert"
EXPERT_EVALUATION_PROFILE = "diagnostic"
SPLITS = ("train", "validation", "test")
_lock = threading.RLock()


def validate_jsonl(path: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    count = 0
    structured: dict[str, Any] | None = None
    expert: dict[str, Any] | None = None
    profile_name: str | None = None
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append({"line": number, "message": f"Invalid JSON: {error.msg}"})
                continue
            message = _validate(example)
            declared_profile = example.get("task_profile") if isinstance(example, dict) else None
            if declared_profile is None and isinstance(example, dict) and example.get("evaluator") == "llm_judge":
                message = message or "evaluator 'llm_judge' requires task_profile 'expert'"
            if profile_name is None and declared_profile is not None:
                if count:
                    message = message or "task_profile must be present on every example"
                elif declared_profile == STRUCTURED_JSON_PROFILE:
                    profile_name = declared_profile
                    structured = _new_structured_metadata()
                elif declared_profile == EXPERT_PROFILE:
                    profile_name = declared_profile
                    expert = _new_expert_metadata()
                else:
                    message = message or "task_profile must be 'structured_json' or 'expert'"
            elif profile_name is not None and declared_profile != profile_name:
                message = message or f"task_profile must be present and equal to {profile_name!r} on every example"
            if message:
                errors.append({"line": number, "message": message})
            else:
                if structured is not None:
                    structured_error = _validate_structured_example(example, structured)
                    if structured_error:
                        errors.append({"line": number, "message": structured_error})
                        continue
                if expert is not None:
                    expert_error = _validate_expert_example(example, expert)
                    if expert_error:
                        errors.append({"line": number, "message": expert_error})
                        continue
                count += 1
    if count == 0:
        errors.append({"line": 1, "message": "Dataset must contain at least one example"})
    if profile_name is None and any("task_profile" in error["message"] for error in errors):
        errors.append({"line": 1, "message": "Task profile is inconsistent"})
    result: dict[str, Any] = {"valid": not errors, "n_examples": count, "errors": errors}
    if structured is not None and not errors:
        result["profile"] = _finish_structured_metadata(structured, count)
    if expert is not None and not errors:
        result["profile"] = _finish_expert_metadata(expert, count)
    return result


def write_dataset_metadata(dataset_id: str, path: Path, validation: dict[str, Any]) -> None:
    profile = validation.get("profile")
    if not profile:
        return
    metadata = {"profile": {**profile, "source_hash": content_hash(path)}}
    directory = _dir(dataset_id)
    temporary = directory / ".metadata.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, directory / "metadata.json")


async def save_uploaded_jsonl(upload: UploadFile) -> tuple[str, Path]:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_id = uuid4().hex
    directory = _dir(dataset_id)
    temporary: Path | None = None
    size = 0
    try:
        with tempfile.NamedTemporaryFile(dir=DATASETS_DIR, delete=False) as file:
            temporary = Path(file.name)
            while chunk := await upload.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("Dataset file exceeds the 500 MB upload limit")
                file.write(chunk)
        directory.mkdir()
        target = directory / "full.jsonl"
        os.replace(temporary, target)
        return dataset_id, target
    except Exception:
        if temporary:
            temporary.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await upload.close()


async def save_predefined_splits(
    train: UploadFile, validation: UploadFile, test: UploadFile
) -> tuple[str | None, dict[str, Any], dict[str, Any] | None]:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_id = uuid4().hex
    staging = Path(tempfile.mkdtemp(prefix=".predefined-", dir=DATASETS_DIR))
    uploads = {"train": train, "validation": validation, "test": test}
    try:
        for name, upload in uploads.items():
            await _save_upload(upload, staging / f"{name}.jsonl")

        validations = {name: validate_jsonl(staging / f"{name}.jsonl") for name in SPLITS}
        errors: list[dict[str, str]] = []
        for name, result in validations.items():
            if not result["valid"]:
                errors.append({"split": name, "message": "JSONL validation failed"})
        group_ids: dict[str, set[str]] = {}
        if not errors:
            for name in SPLITS:
                try:
                    group_ids[name] = _predefined_group_ids(staging / f"{name}.jsonl")
                    validations[name]["n_groups"] = len(group_ids[name])
                except ValueError as error:
                    errors.append({"split": name, "message": str(error)})
        if not errors:
            errors.extend(_predefined_profile_errors(validations))
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                if group_ids[left] & group_ids[right]:
                    errors.append({"split": f"{left}/{right}", "message": "group_id values must not overlap between predefined splits"})
        if errors:
            shutil.rmtree(staging, ignore_errors=True)
            return None, {"valid": False, "splits": validations, "errors": errors}, None

        full = staging / "full.jsonl"
        _concatenate_splits(staging, full)
        full_validation = validate_jsonl(full)
        if not full_validation["valid"]:
            shutil.rmtree(staging, ignore_errors=True)
            return None, {"valid": False, "splits": validations, "errors": [{"split": "dataset", "message": "Combined profile validation failed"}]}, None

        counts = {name: validations[name]["n_examples"] for name in SPLITS}
        metadata: dict[str, Any] = {
            "dataset_id": dataset_id,
            "dataset_mode": "predefined",
            "source_example_count": sum(counts.values()),
            "source_group_count": len(set().union(*group_ids.values())),
            "source_hash": content_hash(full),
            "split": {
                "seed": None,
                **{f"{name}_count": counts[name] for name in SPLITS},
                **{f"{name}_example_count": counts[name] for name in SPLITS},
                **{f"{name}_group_count": len(group_ids[name]) for name in SPLITS},
            },
            **{f"{name}_hash": content_hash(staging / f"{name}.jsonl") for name in SPLITS},
        }
        if full_validation.get("profile"):
            metadata["profile"] = {**full_validation["profile"], "source_hash": metadata["source_hash"]}
        (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        os.replace(staging, _dir(dataset_id))
        return dataset_id, {"valid": True, "n_examples": metadata["source_example_count"], "profile": metadata.get("profile"), "splits": validations}, _public(metadata)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for upload in uploads.values():
            await upload.close()


def dataset_path(dataset_id: str) -> Path:
    _id(dataset_id)
    new_path = _dir(dataset_id) / "full.jsonl"
    old_path = DATASETS_DIR / f"{dataset_id}.jsonl"
    if new_path.is_file():
        return new_path
    if old_path.is_file():
        return old_path
    raise FileNotFoundError(dataset_id)


def create_canonical_split(
    dataset_id: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("Split ratios must be positive and sum to 1")

    with _lock:
        directory = _dir(dataset_id)
        if directory.is_symlink():
            raise ValueError("Dataset symlinks are not allowed")
        metadata_path = directory / "metadata.json"
        existing = _read(metadata_path)
        if existing.get("dataset_mode") == "predefined":
            raise RuntimeError("Split is already predefined by user; canonical split cannot be recreated")
        source = dataset_path(dataset_id)
        if _ready(existing, directory):
            split = existing["split"]
            if all(split[f"{name}_ratio"] == ratios[name] for name in SPLITS) and split["seed"] == seed:
                return _public(existing)
            raise RuntimeError("Canonical split already exists with different ratios or seed")

        validation = validate_jsonl(source)
        if not validation["valid"]:
            raise ValueError("Dataset validation failed")
        total = validation["n_examples"]
        groups = _groups(source)
        if total < 3:
            raise ValueError("Dataset needs at least 3 examples for train, validation, and test")

        if groups is None:
            split_mode = "example"
            target_counts = _counts(total, ratios)
            counts = {name: 0 for name in SPLITS}
            allocation = None
            group_counts = {name: None for name in SPLITS}
        else:
            split_mode = "group"
            if len(groups) < 3:
                raise ValueError("Dataset needs at least 3 unique group_id values for non-empty splits")
            group_counts = _counts(len(groups), ratios)
            group_ids = sorted(groups)
            random.Random(seed).shuffle(group_ids)
            allocation = {}
            start = 0
            for name in SPLITS:
                for group_id in group_ids[start : start + group_counts[name]]:
                    allocation[group_id] = name
                start += group_counts[name]
            counts = {name: 0 for name in SPLITS}
        directory.mkdir(exist_ok=True)
        temporary_paths = {name: directory / f".{name}.tmp" for name in SPLITS}
        targets = {name: directory / f"{name}.jsonl" for name in SPLITS}
        remaining = target_counts.copy() if allocation is None else {}
        remaining_total = total
        generator = random.Random(seed)
        try:
            with (
                source.open(encoding="utf-8") as input_file,
                temporary_paths["train"].open("w", encoding="utf-8") as train_file,
                temporary_paths["validation"].open("w", encoding="utf-8") as validation_file,
                temporary_paths["test"].open("w", encoding="utf-8") as test_file,
            ):
                outputs = {"train": train_file, "validation": validation_file, "test": test_file}
                for line in input_file:
                    if allocation is None:
                        split = _choose_split(generator, remaining, remaining_total)
                        remaining[split] -= 1
                        remaining_total -= 1
                    else:
                        split = allocation[json.loads(line)["group_id"]]
                    outputs[split].write(line)
                    counts[split] += 1
            for name in SPLITS:
                os.replace(temporary_paths[name], targets[name])
            metadata = {
                "dataset_id": dataset_id,
                "dataset_mode": "canonical",
                "split_mode": split_mode,
                "source_example_count": total,
                "source_group_count": len(groups) if groups is not None else None,
                "source_hash": content_hash(source),
                "split": {
                    **{f"{name}_ratio": ratios[name] for name in SPLITS},
                    "seed": seed,
                    **{f"{name}_count": counts[name] for name in SPLITS},
                    **{f"{name}_example_count": counts[name] for name in SPLITS},
                    **{f"{name}_group_count": group_counts[name] for name in SPLITS},
                },
                **{f"{name}_hash": content_hash(targets[name]) for name in SPLITS},
            }
            if validation.get("profile"):
                metadata["profile"] = {
                    **validation["profile"],
                    "source_hash": metadata["source_hash"],
                }
            temporary_metadata = directory / ".metadata.tmp"
            temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary_metadata, metadata_path)
        except Exception:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)
            raise
        return _public(metadata)


def split_path(dataset_id: str, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError("Invalid dataset split")
    directory = _dir(dataset_id)
    metadata = _read(directory / "metadata.json")
    if not _ready(metadata, directory):
        raise RuntimeError("Dataset must be split before training or evaluation")
    return directory / f"{split}.jsonl"


def training_dataset_path(dataset_id: str) -> Path:
    return split_path(dataset_id, "train")


def validation_dataset_path(dataset_id: str) -> Path:
    return split_path(dataset_id, "validation")


def dataset_lineage(dataset_id: str) -> dict[str, Any]:
    directory = _dir(dataset_id)
    metadata = _read(directory / "metadata.json")
    if not _ready(metadata, directory):
        raise RuntimeError("Dataset must be split before training or evaluation")
    return {
        "training_split": "train",
        "dataset_mode": metadata.get("dataset_mode", "canonical"),
        "split_mode": metadata.get("split_mode", "example"),
        "source_hash": metadata["source_hash"],
        "train_hash": metadata["train_hash"],
        "validation_split": "validation",
        "validation_hash": metadata["validation_hash"],
        "test_hash": metadata["test_hash"],
        "validation_count": metadata["split"]["validation_count"],
        "split_seed": metadata["split"].get("seed"),
        "split_config": metadata["split"],
        "source_group_count": metadata.get("source_group_count"),
        "profile": metadata.get("profile"),
    }


def dataset_profile(dataset_id: str) -> dict[str, Any] | None:
    profile = _read(_dir(dataset_id) / "metadata.json").get("profile")
    return profile if isinstance(profile, dict) else None


def preview_jsonl(path: Path, n: int = 10) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for _, line in zip(range(n), file)]


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                canonical = json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                digest.update(canonical.encode("utf-8") + b"\n")
    return digest.hexdigest()


def list_datasets() -> list[dict]:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ids = {path.stem for path in DATASETS_DIR.glob("*.jsonl") if DATASET_ID_PATTERN.fullmatch(path.stem)}
    ids.update(
        path.name for path in DATASETS_DIR.iterdir()
        if path.is_dir() and not path.is_symlink() and DATASET_ID_PATTERN.fullmatch(path.name)
    )
    return [_info(dataset_id) for dataset_id in sorted(ids)]


def delete_dataset(dataset_id: str) -> None:
    with _lock:
        source = dataset_path(dataset_id)
        directory = _dir(dataset_id)
        if directory.is_symlink():
            raise ValueError("Dataset symlinks are not allowed")
        if directory.is_dir():
            shutil.rmtree(directory)
        if source.parent == DATASETS_DIR:
            source.unlink()


def _counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in SPLITS}
    result = {name: int(raw[name]) for name in SPLITS}
    missing = total - sum(result.values())
    for name in sorted(SPLITS, key=lambda item: (-(raw[item] - result[item]), SPLITS.index(item)))[:missing]:
        result[name] += 1
    for name in SPLITS:
        if result[name] == 0:
            donor = max(SPLITS, key=lambda item: result[item])
            if result[donor] <= 1:
                raise ValueError("Dataset is too small for non-empty splits")
            result[donor] -= 1
            result[name] = 1
    return result


def _choose_split(generator: random.Random, remaining: dict[str, int], remaining_total: int) -> str:
    point = generator.randrange(remaining_total)
    for name in SPLITS:
        if point < remaining[name]:
            return name
        point -= remaining[name]
    raise RuntimeError("Could not select a dataset split")


def _info(dataset_id: str) -> dict[str, Any]:
    source = dataset_path(dataset_id)
    metadata = _read(_dir(dataset_id) / "metadata.json")
    if _ready(metadata, _dir(dataset_id)):
        return {"id": dataset_id, "size_bytes": source.stat().st_size, "split_status": "ready", **_public(metadata)}
    return {"id": dataset_id, "size_bytes": source.stat().st_size, "split_status": "not_created"}


def _public(metadata: dict[str, Any]) -> dict[str, Any]:
    result = {
        "dataset_id": metadata["dataset_id"],
        "dataset_mode": metadata.get("dataset_mode", "canonical"),
        "source_count": metadata["source_example_count"],
        "source_group_count": metadata.get("source_group_count"),
        "split": metadata["split"],
        **{key: metadata[key] for key in ("source_hash", "train_hash", "validation_hash", "test_hash")},
    }
    if "split_mode" in metadata:
        result["split_mode"] = metadata["split_mode"]
    if metadata.get("profile"):
        result["profile"] = metadata["profile"]
    return result


def _ready(metadata: dict[str, Any], directory: Path) -> bool:
    return bool(metadata) and all((directory / f"{name}.jsonl").is_file() for name in SPLITS)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


async def _save_upload(upload: UploadFile, target: Path) -> None:
    size = 0
    with target.open("wb") as file:
        while chunk := await upload.read(CHUNK_SIZE):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise ValueError("Dataset file exceeds the 500 MB upload limit")
            file.write(chunk)


def _predefined_group_ids(path: Path) -> set[str]:
    groups: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            group_id = json.loads(line).get("group_id")
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError("predefined split requires a non-empty group_id on every example")
            groups.add(group_id)
    return groups


def _predefined_profile_errors(validations: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    profiles = {name: value.get("profile") for name, value in validations.items()}
    names = {profile.get("task_profile") if isinstance(profile, dict) else None for profile in profiles.values()}
    if len(names) != 1:
        return [{"split": "dataset", "message": "task_profile must be compatible across predefined splits"}]
    profile_name = names.pop()
    if profile_name == STRUCTURED_JSON_PROFILE:
        fields = {tuple(profile["target_schema"]["fields"]) for profile in profiles.values() if profile}
        if len(fields) != 1:
            return [{"split": "dataset", "message": "structured JSON target fields must match across predefined splits"}]
    if profile_name == EXPERT_PROFILE:
        values = {
            (profile.get("task_profile"), profile.get("evaluation_profile"), profile.get("evaluator"))
            for profile in profiles.values()
            if profile
        }
        if values != {(EXPERT_PROFILE, EXPERT_EVALUATION_PROFILE, "llm_judge")}:
            return [{"split": "dataset", "message": "expert profile must match across predefined splits"}]
    return []


def _concatenate_splits(directory: Path, target: Path) -> None:
    with target.open("w", encoding="utf-8") as output:
        for name in SPLITS:
            last_line = ""
            with (directory / f"{name}.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    output.write(line)
                    last_line = line
            if last_line and not last_line.endswith("\n"):
                output.write("\n")


def _dir(dataset_id: str) -> Path:
    _id(dataset_id)
    path = (DATASETS_DIR / dataset_id).resolve()
    try:
        path.relative_to(DATASETS_DIR.resolve())
    except ValueError as error:
        raise ValueError("Dataset path must stay inside the datasets directory") from error
    return path


def _id(dataset_id: str) -> None:
    if not DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError("Invalid dataset id")


def _validate(example: Any) -> str | None:
    if not isinstance(example, dict):
        return "Each line must be a JSON object"
    if "group_id" in example and (
        not isinstance(example["group_id"], str) or not example["group_id"].strip()
    ):
        return "group_id must be a non-empty string when present"
    if "evaluator" in example and (
        not isinstance(example["evaluator"], str) or example["evaluator"] not in ALLOWED_EVALUATORS
    ):
        return "evaluator must be one of: exact, normalized_text, json, llm_judge"
    messages = example.get("messages")
    if not isinstance(messages, list):
        return "messages must be a list"
    if len(messages) < 2:
        return "messages must contain at least 2 items"
    for index, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            return f"message {index} must be an object"
        if message.get("role") not in ALLOWED_ROLES:
            return f"message {index} has an invalid role"
        if not isinstance(message.get("content"), str):
            return f"message {index} content must be a string"
    return None if messages[-1]["role"] == "assistant" else "The last message role must be assistant"


def _new_structured_metadata() -> dict[str, Any]:
    return {
        "fields": None,
        "groups": set(),
        "field_stats": {},
        "total_null_values": 0,
        "examples_with_null": 0,
    }


def _new_expert_metadata() -> dict[str, Any]:
    return {"groups": set(), "evaluation_profile": None}


def _validate_expert_example(example: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    if example.get("evaluator") != "llm_judge":
        return "expert examples must use evaluator 'llm_judge'"
    if example.get("evaluation_profile") != EXPERT_EVALUATION_PROFILE:
        return "expert examples must use evaluation_profile 'diagnostic'"
    if not isinstance(example.get("group_id"), str) or not example["group_id"].strip():
        return "expert examples must have a non-empty group_id"
    target = example["messages"][-1]["content"]
    if not target.strip():
        return "expert assistant content must be a non-empty string"
    if metadata["evaluation_profile"] is None:
        metadata["evaluation_profile"] = example["evaluation_profile"]
    elif metadata["evaluation_profile"] != example["evaluation_profile"]:
        return "expert evaluation_profile must match every example"
    metadata["groups"].add(example["group_id"])
    return None


def _finish_expert_metadata(metadata: dict[str, Any], n_examples: int) -> dict[str, Any]:
    return {
        "task_profile": EXPERT_PROFILE,
        "evaluation_profile": metadata["evaluation_profile"],
        "evaluator": "llm_judge",
        "n_examples": n_examples,
        "n_groups": len(metadata["groups"]),
    }


def _validate_structured_example(example: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    if example.get("evaluator") != "json":
        return "structured JSON examples must use evaluator 'json'"
    if not isinstance(example.get("group_id"), str) or not example["group_id"].strip():
        return "structured JSON examples must have a non-empty group_id"
    try:
        target = json.loads(example["messages"][-1]["content"])
    except json.JSONDecodeError as error:
        return f"structured JSON assistant content is invalid JSON: {error.msg}"
    if not isinstance(target, dict):
        return "structured JSON assistant content must be a top-level JSON object"
    fields = sorted(target)
    if metadata["fields"] is None:
        metadata["fields"] = fields
        metadata["field_stats"] = {
            field: {"observed_types": set(), "null_count": 0} for field in fields
        }
    elif fields != metadata["fields"]:
        return "structured JSON target top-level keys must match every example"

    metadata["groups"].add(example["group_id"])
    has_null = False
    for field in fields:
        value = target[field]
        statistics = metadata["field_stats"][field]
        statistics["observed_types"].add(_json_type(value))
        if value is None:
            statistics["null_count"] += 1
            metadata["total_null_values"] += 1
            has_null = True
    if has_null:
        metadata["examples_with_null"] += 1
    return None


def _finish_structured_metadata(metadata: dict[str, Any], n_examples: int) -> dict[str, Any]:
    fields = metadata["fields"] or []
    return {
        "task_profile": STRUCTURED_JSON_PROFILE,
        "n_examples": n_examples,
        "n_groups": len(metadata["groups"]),
        "target_schema": {
            "fields": fields,
            "field_stats": {
                field: {
                    "observed_types": sorted(metadata["field_stats"][field]["observed_types"]),
                    "null_count": metadata["field_stats"][field]["null_count"],
                    "null_rate": metadata["field_stats"][field]["null_count"] / n_examples,
                }
                for field in fields
            },
        },
        "null_statistics": {
            "total_null_values": metadata["total_null_values"],
            "examples_with_null": metadata["examples_with_null"],
            "examples_without_null": n_examples - metadata["examples_with_null"],
        },
    }


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _groups(path: Path) -> dict[str, int] | None:
    groups: dict[str, int] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            example = json.loads(line)
            group_id = example.get("group_id")
            if group_id is None:
                if groups:
                    raise ValueError("Mixed group_id presence is not allowed")
                groups = None
                for remaining in file:
                    if "group_id" in json.loads(remaining):
                        raise ValueError("Mixed group_id presence is not allowed")
                return None
            if groups is None:
                raise ValueError("Mixed group_id presence is not allowed")
            groups[group_id] = groups.get(group_id, 0) + 1
    return groups
