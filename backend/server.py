import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from adapter_utils import adapter_info, adapter_path, list_adapters
from config import model_path_readiness, settings
from evaluation import evaluation_results, get_evaluation, list_evaluations, reconcile_evaluations, start_evaluation
from dataset_utils import (
    create_canonical_split,
    dataset_path,
    delete_dataset as delete_dataset_storage,
    list_datasets,
    preview_jsonl,
    save_predefined_splits,
    save_uploaded_jsonl,
    validate_jsonl,
    write_dataset_metadata,
)
from inference import inference_manager
from schemas import EvaluationStartRequest, HealthResponse, InferenceRequest, InferenceResponse, SplitRequest
from schemas import TrainingStartRequest
from training import (
    GpuBusyError,
    delete_inactive_adapter,
    get_training,
    is_active_dataset,
    list_training,
    gpu_state,
    start_training,
)


app = FastAPI(title="Qwen LoRA Fine-tuning Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
def reconcile_interrupted_evaluations() -> None:
    reconcile_evaluations()


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, error: HTTPException) -> JSONResponse:
    codes = {
        400: "BAD_REQUEST",
        404: "NOT_FOUND",
        409: "CONFLICT",
        413: "FILE_TOO_LARGE",
        422: "VALIDATION_ERROR",
    }
    content = {
        "error": {
            "code": codes.get(error.status_code, "REQUEST_ERROR"),
            "message": "Dataset validation failed"
            if isinstance(error.detail, dict)
            else str(error.detail),
        }
    }
    if isinstance(error.detail, dict):
        content["validation"] = error.detail
    return JSONResponse(status_code=error.status_code, content=content)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        gpu_state=gpu_state(),
        model_loaded=bool(inference_manager.status()["loaded"]),
        **model_path_readiness(),
    )


@app.post("/api/datasets/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are accepted")

    try:
        dataset_id, path = await save_uploaded_jsonl(file)
    except ValueError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error

    validation = validate_jsonl(path)
    if not validation["valid"]:
        delete_dataset_storage(dataset_id)
        raise HTTPException(status_code=422, detail=validation)
    write_dataset_metadata(dataset_id, path, validation)
    return {"id": dataset_id, "validation": validation}


@app.post("/api/datasets/upload-predefined", status_code=status.HTTP_201_CREATED)
async def upload_predefined_dataset(
    train: UploadFile = File(...), validation: UploadFile = File(...), test: UploadFile = File(...)
) -> dict:
    uploads = {"train": train, "validation": validation, "test": test}
    if any(not upload.filename or Path(upload.filename).suffix.lower() != ".jsonl" for upload in uploads.values()):
        raise HTTPException(status_code=400, detail="Train, validation, and test must be .jsonl files")
    try:
        dataset_id, result, dataset = await save_predefined_splits(train, validation, test)
    except ValueError as error:
        status_code = 413 if "500 MB" in str(error) else 400
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    if not dataset_id or not dataset:
        raise HTTPException(status_code=422, detail=result)
    return {"id": dataset_id, "validation": result, "splits": result["splits"], "dataset": dataset}


@app.get("/api/datasets")
def get_datasets() -> list[dict]:
    return list_datasets()


@app.get("/api/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str, n: int = Query(default=10, ge=1, le=100)) -> dict:
    path = _get_dataset_path(dataset_id)
    return {"id": dataset_id, "examples": preview_jsonl(path, n)}


@app.post("/api/datasets/{dataset_id}/split")
def split_dataset(dataset_id: str, request: SplitRequest) -> dict:
    try:
        return create_canonical_split(
            dataset_id,
            request.train_ratio,
            request.validation_ratio,
            request.test_ratio,
            request.seed,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Dataset not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.delete("/api/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str) -> None:
    _get_dataset_path(dataset_id)
    if is_active_dataset(dataset_id):
        raise HTTPException(status_code=409, detail="Dataset is used by an active training job")
    try:
        delete_dataset_storage(dataset_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/training/start", status_code=status.HTTP_201_CREATED)
def start_training_job(request: TrainingStartRequest) -> dict:
    config = request.model_dump(exclude={"dataset_id", "job_id"})
    try:
        return start_training(request.dataset_id, request.job_id, config)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Dataset not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/training")
def list_training_jobs() -> list[dict]:
    return list_training()


@app.get("/training/{job_id}")
def get_training_job(job_id: str) -> dict:
    job = get_training(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@app.post("/evaluations", status_code=status.HTTP_201_CREATED)
def create_evaluation(request: EvaluationStartRequest) -> dict:
    try:
        return start_evaluation(request.dataset_id, request.adapter_id, request.max_new_tokens)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/evaluations")
def get_evaluations() -> list[dict]:
    return list_evaluations()


@app.get("/evaluations/{evaluation_id}")
def get_evaluation_job(evaluation_id: str) -> dict:
    try:
        evaluation = get_evaluation(evaluation_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


@app.get("/evaluations/{evaluation_id}/results")
def get_evaluation_results(evaluation_id: str) -> list[dict]:
    try:
        return evaluation_results(evaluation_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Evaluation not found") from error


@app.get("/inference/status")
def inference_status() -> dict:
    return inference_manager.status()


@app.post("/inference/generate", response_model=InferenceResponse)
def generate_inference(request: InferenceRequest) -> dict:
    try:
        if request.messages is not None:
            response = inference_manager.generate_messages(
                [message.model_dump() for message in request.messages],
                request.adapter_id,
                request.max_new_tokens,
                request.temperature,
            )
        else:
            response = inference_manager.generate(
                request.prompt or "",
                request.adapter_id,
                request.max_new_tokens,
                request.temperature,
            )
        try:
            parsed_json = json.loads(str(response["text"]))
            json_valid = True
        except json.JSONDecodeError:
            parsed_json = None
            json_valid = False
        return {**response, "json_valid": json_valid, "parsed_json": parsed_json}
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except GpuBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error


@app.get("/adapters")
def get_adapters() -> list[dict]:
    return list_adapters()


@app.get("/adapters/{adapter_id}")
def get_adapter(adapter_id: str) -> dict:
    try:
        return adapter_info(adapter_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Adapter not found") from error


@app.get("/adapters/{adapter_id}/download")
def download_adapter(adapter_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    try:
        path = adapter_path(adapter_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Adapter not found") from error
    descriptor, temporary_name = tempfile.mkstemp(suffix=".zip")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in path.iterdir():
                if item.is_file() and not item.is_symlink() and _downloadable_adapter_file(item.name):
                    archive.write(item, item.name)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    background_tasks.add_task(_remove_temporary_download, temporary)
    return FileResponse(temporary, media_type="application/zip", filename=f"{adapter_id}.zip", background=background_tasks)


@app.delete("/adapters/{adapter_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_adapter(adapter_id: str) -> None:
    try:
        path = adapter_path(adapter_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Adapter not found") from error
    try:
        delete_inactive_adapter(adapter_id, lambda: inference_manager.delete_adapter(adapter_id, path))
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _get_dataset_path(dataset_id: str) -> Path:
    try:
        return dataset_path(dataset_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Dataset not found") from error


def _downloadable_adapter_file(name: str) -> bool:
    return name in {"adapter_model.safetensors", "adapter_model.bin", "adapter_config.json", "metadata.json", "README.md"} or name.startswith("tokenizer_") or name in {"tokenizer.json", "special_tokens_map.json"}


def _remove_temporary_download(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary adapter download %s", path)
