"""Safe local adapter storage helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parent
ADAPTERS_DIR = BACKEND_DIR / "data" / "adapters"
ADAPTER_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
REQUIRED_MODEL_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def adapter_path(adapter_id: str) -> Path:
    if not ADAPTER_ID_PATTERN.fullmatch(adapter_id):
        raise ValueError("Invalid adapter id")
    candidate = ADAPTERS_DIR / adapter_id
    if candidate.is_symlink():
        raise ValueError("Adapter symlinks are not allowed")
    path = candidate.resolve()
    try:
        path.relative_to(ADAPTERS_DIR.resolve())
    except ValueError as error:
        raise ValueError("Adapter path must stay inside the adapters directory") from error
    if not path.is_dir():
        raise FileNotFoundError("Adapter not found")
    return path


def adapter_validation(path: Path) -> tuple[bool, str | None]:
    if not (path / "adapter_config.json").is_file():
        return False, "Missing adapter_config.json"
    if not any((path / filename).is_file() for filename in REQUIRED_MODEL_FILES):
        return False, "Missing adapter model file"
    return True, None


def list_adapters() -> list[dict[str, Any]]:
    if not ADAPTERS_DIR.is_dir():
        return []
    adapters = []
    for entry in sorted(ADAPTERS_DIR.iterdir(), key=lambda path: path.name):
        if not entry.is_dir() or entry.is_symlink() or not ADAPTER_ID_PATTERN.fullmatch(entry.name):
            continue
        try:
            adapters.append(adapter_info(entry.name, include_files=False))
        except (FileNotFoundError, ValueError):
            continue
    return adapters


def adapter_info(adapter_id: str, include_files: bool = True) -> dict[str, Any]:
    path = adapter_path(adapter_id)
    valid, validation_error = adapter_validation(path)
    metadata, metadata_error = _read_metadata(path)
    created_at, created_at_source = _created_at(path, metadata)
    info: dict[str, Any] = {
        "adapter_id": adapter_id,
        "valid": valid,
        "validation_error": validation_error,
        "metadata_available": metadata is not None,
        "metadata_error": metadata_error,
        "created_at": created_at,
        "created_at_source": created_at_source,
        "size_bytes": _size_bytes(path),
    }
    if include_files:
        info["files"] = _files(path)
        info["training_job_id"] = metadata.get("training_job_id") if metadata else None
        info["dataset_id"] = metadata.get("dataset_id") if metadata else None
        info["dataset_lineage"] = metadata.get("dataset_lineage") if metadata else None
        info["training_config"] = metadata.get("training_config") if metadata else None
        info["best_epoch"] = metadata.get("best_epoch") if metadata else None
        info["best_eval_loss"] = metadata.get("best_eval_loss") if metadata else None
        info["best_checkpoint"] = metadata.get("best_checkpoint") if metadata else None
    return info


def write_adapter_metadata(job: dict[str, Any]) -> None:
    adapter_id = job["job_id"]
    path = adapter_path(adapter_id)
    metadata = {
        "adapter_id": adapter_id,
        "training_job_id": job["job_id"],
        "dataset_id": job["dataset"],
        "dataset_lineage": job.get("dataset_lineage"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_config": {key: value for key, value in job["config"].items() if key != "exp_id"},
        "best_epoch": job.get("best_epoch"),
        "best_eval_loss": job.get("best_eval_loss"),
        "best_checkpoint": job.get("best_checkpoint"),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _read_metadata(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return None, None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "Invalid metadata.json"
    return (data, None) if isinstance(data, dict) else (None, "Invalid metadata.json")


def _created_at(path: Path, metadata: dict[str, Any] | None) -> tuple[str, str]:
    created_at = metadata.get("created_at") if metadata else None
    if isinstance(created_at, str):
        return created_at, "metadata"
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "filesystem"


def _files(path: Path) -> list[str]:
    return sorted(
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())
