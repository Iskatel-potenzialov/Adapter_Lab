"""Subprocess-based training jobs for the local backend process."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import settings
from dataset_utils import dataset_lineage, training_dataset_path, validation_dataset_path


BACKEND_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = BACKEND_DIR / "train.py"
ADAPTERS_DIR = BACKEND_DIR / "data" / "adapters"
EXPERIMENTS_DIR = BACKEND_DIR / "data" / "experiments"
LOGS_DIR = BACKEND_DIR / "logs"
JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
MAX_EVENTS = 100

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()
_gpu_lock = threading.RLock()
_gpu_owner: str | None = None


class GpuBusyError(RuntimeError):
    pass


def start_training(dataset_id: str, job_id: str | None, config: dict[str, Any]) -> dict[str, Any]:
    source_path = training_dataset_path(dataset_id)
    validation_path = validation_dataset_path(dataset_id)
    lineage = dataset_lineage(dataset_id)
    job_id = job_id or uuid4().hex
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid job id")

    adapter_path = _child_path(ADAPTERS_DIR, job_id)
    config_path = _child_path(EXPERIMENTS_DIR, f"{job_id}_config.json")
    log_path = _child_path(LOGS_DIR, f"{job_id}.log")
    with _lock:
        if any(job["status"] in {"queued", "running"} for job in _jobs.values()):
            raise RuntimeError("A training job is already running")
        if job_id in _jobs:
            raise RuntimeError("Training job id already exists")
        if adapter_path.exists():
            raise RuntimeError("Adapter output already exists")
        if config_path.exists():
            raise RuntimeError("Training config already exists")

        _reserve_training()
        try:
            from inference import inference_manager

            inference_manager.unload_for_training()
        except Exception:
            _release_training()
            raise
        try:
            EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
            config = {"exp_id": job_id, **config}
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        except Exception:
            _release_training()
            raise
        job = {
            "job_id": job_id,
            "status": "queued",
            "dataset": dataset_id,
            "dataset_lineage": lineage,
            "output": str(adapter_path),
            "config": config,
            "log_path": str(log_path),
            "pid": None,
            "created_at": _timestamp(),
            "started_at": None,
            "finished_at": None,
            "progress": None,
            "current_step": None,
            "loss": None,
            "learning_rate": None,
            "epoch": None,
            "eval_loss": None,
            "eval_runtime": None,
            "best_epoch": None,
            "best_eval_loss": None,
            "best_checkpoint": None,
            "error": None,
            "adapter_path": None,
            "metadata_error": None,
            "events": [],
            "_done": False,
        }
        environment = os.environ.copy()
        if settings.model_path:
            environment["MODEL_PATH"] = settings.model_path
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--dataset",
            str(source_path),
            "--validation",
            str(validation_path),
            "--output",
            str(adapter_path),
            "--config",
            str(config_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(BACKEND_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
        except OSError as error:
            config_path.unlink(missing_ok=True)
            _release_training()
            raise RuntimeError(f"Could not start training process: {error}") from error

        job["pid"] = process.pid
        job["_process"] = process
        _jobs[job_id] = job
        threading.Thread(target=_read_process, args=(job_id,), daemon=True).start()
        return _public_job(job)


def get_training(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return _public_job(job) if job else None


def list_training() -> list[dict[str, Any]]:
    with _lock:
        return [_public_job(job) for job in _jobs.values()]


def is_active_adapter_output(adapter_id: str) -> bool:
    with _lock:
        return any(
            job["status"] in {"queued", "running"} and Path(job["output"]).name == adapter_id
            for job in _jobs.values()
        )


def is_active_dataset(dataset_id: str) -> bool:
    with _lock:
        return any(
            job["status"] in {"queued", "running"} and job["dataset"] == dataset_id
            for job in _jobs.values()
        )


def delete_inactive_adapter(adapter_id: str, delete: Any) -> None:
    with _lock:
        if is_active_adapter_output(adapter_id):
            raise RuntimeError("Adapter is being created by an active training job")
        delete()


def has_active_training() -> bool:
    with _lock:
        return any(job["status"] in {"queued", "running"} for job in _jobs.values())


def reserve_inference() -> None:
    global _gpu_owner
    with _gpu_lock:
        if _gpu_owner:
            raise GpuBusyError(f"GPU is busy with {_gpu_owner}")
        _gpu_owner = "inference"


def release_inference() -> None:
    global _gpu_owner
    with _gpu_lock:
        if _gpu_owner == "inference":
            _gpu_owner = None


def _reserve_training() -> None:
    global _gpu_owner
    with _gpu_lock:
        if _gpu_owner:
            raise GpuBusyError(f"GPU is busy with {_gpu_owner}")
        _gpu_owner = "training"


def _release_training() -> None:
    global _gpu_owner
    with _gpu_lock:
        if _gpu_owner == "training":
            _gpu_owner = None


def reserve_evaluation() -> None:
    global _gpu_owner
    with _gpu_lock:
        if _gpu_owner:
            raise GpuBusyError(f"GPU is busy with {_gpu_owner}")
        _gpu_owner = "evaluation"


def release_evaluation() -> None:
    global _gpu_owner
    with _gpu_lock:
        if _gpu_owner == "evaluation":
            _gpu_owner = None


def gpu_state() -> str:
    with _gpu_lock:
        return _gpu_owner or "idle"


def _read_process(job_id: str) -> None:
    try:
        with _lock:
            job = _jobs.get(job_id)
            if not job:
                return
            process = job["_process"]
            job["status"] = "running"
            job["started_at"] = _timestamp()

        log_path = Path(job["log_path"])
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            if process.stdout:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    _handle_output(job_id, line.rstrip())

        return_code = process.wait()
        with _lock:
            job = _jobs.get(job_id)
            if not job:
                return
            if return_code != 0:
                job["status"] = "failed"
                job["error"] = job["error"] or f"Training process exited with code {return_code}"
            elif job["_done"]:
                job["status"] = "completed"
                try:
                    from adapter_utils import write_adapter_metadata

                    write_adapter_metadata(job)
                except Exception as error:
                    job["metadata_error"] = f"Could not write adapter metadata: {error}"
            else:
                job["status"] = "failed"
                job["error"] = "Training process exited without a done event"
            job["finished_at"] = _timestamp()
    except Exception as error:
        with _lock:
            job = _jobs.get(job_id)
            if job:
                job["status"] = "failed"
                job["error"] = f"Training output reader failed: {error}"
                job["finished_at"] = _timestamp()
    finally:
        _release_training()


def _handle_output(job_id: str, line: str) -> None:
    if not line:
        return
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        event = {"event": "raw", "line": line}

    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["events"].append(event)
        if len(job["events"]) > MAX_EVENTS:
            job["events"].pop(0)

        event_name = event.get("event") if isinstance(event, dict) else None
        if event_name == "started":
            job["status"] = "running"
            job["started_at"] = job["started_at"] or _timestamp()
        elif event_name == "metrics":
            job["current_step"] = event.get("step")
            if event.get("loss") is not None:
                job["loss"] = event["loss"]
            if event.get("lr") is not None:
                job["learning_rate"] = event["lr"]
            if event.get("epoch") is not None:
                job["epoch"] = event["epoch"]
            if event.get("eval_loss") is not None:
                job["eval_loss"] = event["eval_loss"]
            if event.get("eval_runtime") is not None:
                job["eval_runtime"] = event["eval_runtime"]
        elif event_name == "done":
            job["_done"] = True
            job["adapter_path"] = job["output"]
            for field in ("best_epoch", "best_eval_loss", "best_checkpoint"):
                if field in event:
                    job[field] = event[field]
        elif event_name == "error":
            job["error"] = event.get("message") or "Training process reported an error"


def _child_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Path must stay inside the project storage directory") from error
    return path


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
