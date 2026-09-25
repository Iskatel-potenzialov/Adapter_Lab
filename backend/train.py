"""Standalone QLoRA training entry point for text-only JSONL datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config import settings
from dataset_utils import validate_jsonl


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}), flush=True)


@dataclass(frozen=True)
class TrainingConfig:
    exp_id: str
    epochs: float
    batch_size: int
    eval_batch_size: int
    grad_accum: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    max_seq_length: int
    early_stopping_patience: int | None
    early_stopping_threshold: float


def read_config(path: Path) -> TrainingConfig:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Config file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid config JSON: {error.msg}") from error

    if not isinstance(values, dict):
        raise ValueError("Config must be a JSON object")

    target_modules = values.get("target_modules", DEFAULT_TARGET_MODULES)
    if not isinstance(target_modules, list) or not all(
        isinstance(item, str) and item for item in target_modules
    ):
        raise ValueError("target_modules must be a non-empty list of strings")

    config = TrainingConfig(
        exp_id=str(values.get("exp_id", path.stem)),
        epochs=float(values.get("epochs", 3)),
        batch_size=int(values.get("batch_size", 1)),
        eval_batch_size=int(values.get("eval_batch_size", 1)),
        grad_accum=int(values.get("grad_accum", 16)),
        learning_rate=float(values.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(values.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(values.get("warmup_ratio", 0.03)),
        lora_r=int(values.get("lora_r", 16)),
        lora_alpha=int(values.get("lora_alpha", 32)),
        lora_dropout=float(values.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        max_seq_length=int(values.get("max_seq_length", 256)),
        early_stopping_patience=(
            None
            if values.get("early_stopping_patience") is None
            else int(values["early_stopping_patience"])
        ),
        early_stopping_threshold=float(values.get("early_stopping_threshold", 0)),
    )
    if not config.exp_id or config.epochs <= 0 or config.batch_size <= 0 or config.eval_batch_size <= 0 or config.grad_accum <= 0:
        raise ValueError("epochs, batch_size, eval_batch_size, grad_accum, and exp_id must be positive")
    if config.learning_rate <= 0 or not 0 <= config.warmup_ratio <= 1:
        raise ValueError("learning_rate must be positive and warmup_ratio must be between 0 and 1")
    if config.lora_r <= 0 or config.lora_alpha <= 0 or not 0 <= config.lora_dropout < 1:
        raise ValueError("Invalid LoRA configuration")
    if config.max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1 when enabled")
    if config.early_stopping_threshold < 0:
        raise ValueError("early_stopping_threshold must be non-negative")
    return config


def format_messages(
    messages: list[dict[str, str]],
    apply_chat_template: Callable[..., str],
) -> str:
    return apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def prompt_completion(messages: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("Training example must end with an assistant message")
    if not isinstance(messages[-1].get("content"), str):
        raise ValueError("Training assistant content must be a string")
    return {"prompt": messages[:-1], "completion": [messages[-1]]}


def prepare_dataset(dataset: Any) -> Any:
    return dataset.map(
        lambda example: prompt_completion(example["messages"]),
        remove_columns=dataset.column_names,
    )


def emit_metrics(logs: dict[str, Any], step: int, epoch: float | None) -> None:
    if "loss" in logs or "eval_loss" in logs:
        emit(
            "metrics",
            step=step,
            loss=logs.get("loss"),
            lr=logs.get("learning_rate"),
            epoch=logs.get("epoch", epoch),
            eval_loss=logs.get("eval_loss"),
            eval_runtime=logs.get("eval_runtime"),
        )


def best_model_summary(trainer: Any) -> dict[str, Any]:
    state = trainer.state
    best_metric = getattr(state, "best_metric", None)
    best_checkpoint = getattr(state, "best_model_checkpoint", None)
    evaluation_logs = [log for log in state.log_history if "eval_loss" in log]
    best_epoch = None
    checkpoint_name = Path(best_checkpoint).name if best_checkpoint else ""
    checkpoint_step = checkpoint_name.removeprefix("checkpoint-")
    if checkpoint_step.isdigit():
        for index, log in enumerate(evaluation_logs, 1):
            if log.get("step") == int(checkpoint_step):
                best_epoch = index
                break
    if best_epoch is None and best_metric is not None:
        for index, log in enumerate(evaluation_logs, 1):
            if log.get("eval_loss") == best_metric:
                best_epoch = index
                break
    return {
        "best_epoch": best_epoch,
        "best_eval_loss": best_metric,
        "best_checkpoint": best_checkpoint,
    }


def emit_lora_diagnostics(model: Any, target_modules: list[str]) -> None:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    emit(
        "model_parameters",
        total=total_parameters,
        trainable=trainable_parameters,
        trainable_percent=round(trainable_parameters * 100 / total_parameters, 6)
        if total_parameters
        else 0,
    )

    peft_base_model = getattr(model, "base_model", None)
    wrapped_model = getattr(peft_base_model, "model", None)
    language_root = getattr(wrapped_model, "model", None)
    vision_root = getattr(wrapped_model, "visual", None)
    language_module_ids = {id(module) for module in language_root.modules()} if language_root else set()
    vision_module_ids = {id(module) for module in vision_root.modules()} if vision_root else set()

    components: dict[str, dict[str, Any]] = {
        "language": {"count": 0, "module_names": set(), "sample_modules": []},
        "vision": {"count": 0, "module_names": set(), "sample_modules": []},
        "other": {"count": 0, "module_names": set(), "sample_modules": []},
    }
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if id(module) in language_module_ids:
            component = "language"
        elif id(module) in vision_module_ids:
            component = "vision"
        else:
            component = "other"
        summary = components[component]
        summary["count"] += 1
        summary["module_names"].add(name.rsplit(".", 1)[-1])
        if len(summary["sample_modules"]) < 5:
            summary["sample_modules"].append(name)

    language_names = components["language"]["module_names"]
    vision_names = components["vision"]["module_names"]
    emit(
        "lora_modules",
        target_modules=target_modules,
        classification="Qwen2.5-VL model subtree membership after PEFT wrapping",
        component_roots_found={"language": bool(language_root), "vision": bool(vision_root)},
        components={
            component: {
                "count": summary["count"],
                "module_names": sorted(summary["module_names"]),
                "sample_modules": summary["sample_modules"],
            }
            for component, summary in components.items()
        },
        shared_language_vision_module_names=sorted(language_names & vision_names),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a text-only QLoRA adapter")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def run(dataset_path: Path, output_path: Path, config_path: Path, validation_path: Path | None = None) -> None:
    config = read_config(config_path)
    if not dataset_path.is_file():
        raise ValueError(f"Dataset file not found: {dataset_path}")

    validation = validate_jsonl(dataset_path)
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['errors']}")
    if validation_path is not None:
        if not validation_path.is_file():
            raise ValueError(f"Validation dataset file not found: {validation_path}")
        validation = validate_jsonl(validation_path)
        if not validation["valid"]:
            raise ValueError(f"Validation dataset validation failed: {validation['errors']}")

    model_path = settings.model_path
    if not model_path or not Path(model_path).is_dir():
        raise ValueError("MODEL_PATH must point to a local model directory")

    os.environ["HF_HUB_OFFLINE"] = "1"
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    import torch

    emit("started", exp_id=config.exp_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
        ),
    )
    emit_lora_diagnostics(model, config.target_modules)

    train_dataset = prepare_dataset(load_dataset("json", data_files=str(dataset_path), split="train"))
    validation_dataset = (
        prepare_dataset(load_dataset("json", data_files=str(validation_path), split="train"))
        if validation_path is not None
        else None
    )

    class JsonMetricsCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> Any:
            emit_metrics(logs or {}, state.global_step, getattr(state, "epoch", None))
            return control

    best_model_args: dict[str, Any] = {}
    if validation_dataset is not None:
        best_model_args.update(
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    callbacks: list[Any] = [JsonMetricsCallback()]
    if validation_dataset is not None and config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_path),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=config.grad_accum,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_ratio=config.warmup_ratio,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_length=config.max_seq_length,
            completion_only_loss=True,
            packing=False,
            logging_steps=1,
            eval_strategy="epoch" if validation_dataset is not None else "no",
            save_strategy="epoch" if validation_dataset is not None else "no",
            report_to="none",
            disable_tqdm=True,
            **best_model_args,
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    model.save_pretrained(output_path)
    emit("done", exp_id=config.exp_id, output=str(output_path), **best_model_summary(trainer))


def main() -> int:
    parser = build_parser()
    try:
        arguments = parser.parse_args()
    except SystemExit as error:
        if error.code:
            emit("error", message="Invalid command line arguments")
        return int(error.code)

    try:
        run(arguments.dataset, arguments.output, arguments.config, arguments.validation)
    except Exception as error:
        emit("error", message=str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
