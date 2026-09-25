from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    gpu_state: Literal["idle", "inference", "training", "evaluation"] = "idle"
    model_path_configured: bool
    model_path_exists: bool
    model_loaded: bool


class SplitRequest(BaseModel):
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    validation_ratio: float = Field(default=0.1, gt=0, lt=1)
    test_ratio: float = Field(default=0.1, gt=0, lt=1)
    seed: int = 42


class TrainingStartRequest(BaseModel):
    dataset_id: str
    job_id: str | None = None
    epochs: float = Field(default=3, gt=0)
    batch_size: int = Field(default=1, gt=0)
    eval_batch_size: int = Field(default=1, gt=0)
    grad_accum: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"], min_length=1
    )
    max_seq_length: int = Field(default=256, gt=0)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    early_stopping_threshold: float = Field(default=0, ge=0)


class InferenceMessage(BaseModel):
    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class InferenceRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=1, max_length=12000)
    messages: list[InferenceMessage] | None = Field(default=None, min_length=1)
    adapter_id: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )
    max_new_tokens: int = Field(default=128, gt=0, le=512)
    temperature: float = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def require_one_input(self) -> "InferenceRequest":
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Provide exactly one of prompt or messages")
        return self


class InferenceResponse(BaseModel):
    text: str
    adapter_id: str | None
    json_valid: bool
    parsed_json: Any | None


class EvaluationStartRequest(BaseModel):
    dataset_id: str
    adapter_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    max_new_tokens: int = Field(default=128, gt=0, le=512)
