"""Text-only local inference for the base Qwen model and LoRA adapters."""

from __future__ import annotations

import gc
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from adapter_utils import adapter_path, adapter_validation
from config import settings
from training import release_inference, reserve_inference


class InferenceManager:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._adapter_id: str | None = None
        self._generating = False
        self._lock = threading.RLock()

    def status(self) -> dict[str, bool | str | None]:
        with self._lock:
            return {
                "loaded": self._model is not None,
                "adapter_id": self._adapter_id,
                "generating": self._generating,
            }

    def generate(
        self,
        prompt: str,
        adapter_id: str | None,
        max_new_tokens: int,
        temperature: float,
    ) -> dict[str, str | int | None]:
        return self._generate([{"role": "user", "content": prompt}], adapter_id, max_new_tokens, temperature, True)

    def generate_messages(
        self, messages: list[dict[str, str]], adapter_id: str | None, max_new_tokens: int, temperature: float
    ) -> dict[str, str | int | None]:
        return self._generate(messages, adapter_id, max_new_tokens, temperature, False)

    def _generate(
        self, messages: list[dict[str, str]], adapter_id: str | None, max_new_tokens: int, temperature: float, reserve: bool
    ) -> dict[str, str | int | None]:
        with self._lock:
            if reserve:
                reserve_inference()
            try:
                self._generating = True
                try:
                    self._load_configuration(adapter_id)
                    assert self._model is not None
                    assert self._tokenizer is not None

                    prompt_text = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = self._tokenizer(prompt_text, return_tensors="pt")
                    inputs = inputs.to(_model_device(self._model))
                    generation_options: dict[str, Any] = {"max_new_tokens": max_new_tokens}
                    if temperature == 0:
                        generation_options["do_sample"] = False
                    else:
                        generation_options.update(do_sample=True, temperature=temperature)
                    output_ids = self._model.generate(**inputs, **generation_options)
                    new_token_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
                    text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
                    return {"text": text, "adapter_id": adapter_id}
                except Exception as error:
                    if _is_cuda_oom(error):
                        self._unload()
                        raise RuntimeError("CUDA out of memory during inference") from error
                    if self._model is None:
                        self._release_memory()
                    raise
                finally:
                    self._generating = False
            finally:
                if reserve:
                    release_inference()

    def unload_for_training(self) -> None:
        with self._lock:
            self._unload()

    def delete_adapter(self, adapter_id: str, path: Path) -> None:
        with self._lock:
            if self._adapter_id == adapter_id:
                raise RuntimeError("Adapter is currently loaded in inference")
            shutil.rmtree(path)

    def _load_configuration(self, adapter_id: str | None) -> None:
        if self._model is not None and self._adapter_id == adapter_id:
            return

        model_path = Path(settings.model_path)
        if not settings.model_path or not model_path.is_dir():
            raise ValueError("MODEL_PATH must point to a local model directory")
        selected_adapter_path = adapter_path(adapter_id) if adapter_id else None
        if selected_adapter_path:
            valid, validation_error = adapter_validation(selected_adapter_path)
            if not valid:
                raise ValueError(f"Invalid PEFT adapter files: {validation_error}")

        self._unload()
        os.environ["HF_HUB_OFFLINE"] = "1"
        model: Any | None = None
        tokenizer: Any | None = None
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                str(model_path),
                local_files_only=True,
                quantization_config=quantization,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            if selected_adapter_path:
                model = PeftModel.from_pretrained(model, str(selected_adapter_path))
            model.eval()
        except Exception:
            del model
            del tokenizer
            self._release_memory()
            raise

        self._model = model
        self._tokenizer = tokenizer
        self._adapter_id = adapter_id

    def _unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._adapter_id = None
        self._release_memory()

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (AttributeError, ImportError):
            pass


def _model_device(model: Any) -> Any:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is not None:
            return weight.device
    device = getattr(model, "device", None)
    return device if device is not None else next(model.parameters()).device


def _is_cuda_oom(error: Exception) -> bool:
    try:
        import torch

        return isinstance(error, torch.cuda.OutOfMemoryError)
    except (AttributeError, ImportError):
        return False


inference_manager = InferenceManager()
