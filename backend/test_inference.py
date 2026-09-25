import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import inference
import adapter_utils
import config
import training
from server import app


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=types.SimpleNamespace(shape=(1, 3)))
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    pad_token = "pad"
    eos_token = "eos"

    def __init__(self) -> None:
        self.messages = None
        self.decoded_ids = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "formatted prompt"

    def __call__(self, text, return_tensors):
        self.input_text = text
        self.return_tensors = return_tensors
        return FakeInputs()

    def decode(self, ids, **kwargs):
        self.decoded_ids = ids
        self.decode_kwargs = kwargs
        return "answer"


class FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [[1, 2, 3, 4, 5]]

    def eval(self):
        return self


class InferenceManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        training._gpu_owner = None
        self.manager = inference.InferenceManager()

    def tearDown(self) -> None:
        training._gpu_owner = None

    def test_status_is_unloaded_without_model_load(self) -> None:
        self.assertEqual(
            self.manager.status(),
            {"loaded": False, "adapter_id": None, "generating": False},
        )

    def test_generation_uses_chat_template_and_decodes_only_new_tokens(self) -> None:
        model = FakeModel()
        tokenizer = FakeTokenizer()
        self.manager._model = model
        self.manager._tokenizer = tokenizer

        response = self.manager.generate("Question", None, 12, 0)

        self.assertEqual(response, {"text": "answer", "adapter_id": None})
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Question"}])
        self.assertEqual(tokenizer.template_kwargs, {"tokenize": False, "add_generation_prompt": True})
        self.assertEqual(tokenizer.decoded_ids, [4, 5])
        self.assertEqual(model.calls[0]["do_sample"], False)
        self.assertNotIn("temperature", model.calls[0])

    def test_sampled_generation_passes_temperature(self) -> None:
        model = FakeModel()
        self.manager._model = model
        self.manager._tokenizer = FakeTokenizer()

        self.manager.generate("Question", None, 12, 0.7)

        self.assertEqual(model.calls[0]["do_sample"], True)
        self.assertEqual(model.calls[0]["temperature"], 0.7)

    def test_same_adapter_is_not_reloaded_and_switching_reloads(self) -> None:
        loads = []
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_transformers = types.ModuleType("transformers")

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                loads.append(("tokenizer", args, kwargs))
                return FakeTokenizer()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                loads.append(("model", args, kwargs))
                return FakeModel()

        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoModelForImageTextToText = FakeAutoModel
        fake_transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = types.SimpleNamespace(from_pretrained=lambda model, path: model)

        with TemporaryDirectory() as directory, patch.object(adapter_utils, "ADAPTERS_DIR", Path(directory)):
            model_path = Path(directory) / "model"
            model_path.mkdir()
            for adapter in ("adapter_a", "adapter_b"):
                path = Path(directory) / adapter
                path.mkdir()
                (path / "adapter_config.json").write_text("{}")
                (path / "adapter_model.safetensors").write_text("model")
            with patch.object(inference, "settings", types.SimpleNamespace(model_path=str(model_path))), patch.dict(
                sys.modules, {"torch": fake_torch, "transformers": fake_transformers, "peft": fake_peft}
            ):
                self.manager._load_configuration(None)
                self.manager._load_configuration(None)
                self.manager._load_configuration("adapter_a")
                self.manager._load_configuration("adapter_a")
                self.manager._load_configuration("adapter_b")

        self.assertEqual([entry[0] for entry in loads].count("model"), 3)
        self.assertEqual(self.manager.status()["adapter_id"], "adapter_b")

    def test_missing_or_unsafe_adapter_is_rejected(self) -> None:
        with TemporaryDirectory() as directory, patch.object(adapter_utils, "ADAPTERS_DIR", Path(directory)):
            with self.assertRaises(FileNotFoundError):
                adapter_utils.adapter_path("missing")
            with self.assertRaises(ValueError):
                adapter_utils.adapter_path("../escape")
            invalid = Path(directory) / "invalid"
            invalid.mkdir()
            (invalid / "adapter_config.json").write_text("{}")
            self.assertFalse(adapter_utils.adapter_validation(adapter_utils.adapter_path("invalid"))[0])

    def test_missing_model_path_is_rejected(self) -> None:
        with patch.object(inference, "settings", types.SimpleNamespace(model_path="")):
            with self.assertRaisesRegex(ValueError, "MODEL_PATH"):
                self.manager._load_configuration(None)

    def test_loading_error_leaves_manager_unloaded(self) -> None:
        with TemporaryDirectory() as directory, patch.object(
            inference, "settings", types.SimpleNamespace(model_path=directory)
        ), patch.dict(sys.modules, {"transformers": types.ModuleType("transformers")}):
            with self.assertRaises(ImportError):
                self.manager._load_configuration(None)
        self.assertFalse(self.manager.status()["loaded"])

    def test_failed_adapter_switch_leaves_manager_unloaded(self) -> None:
        self.manager._model = FakeModel()
        self.manager._tokenizer = FakeTokenizer()
        self.manager._adapter_id = "adapter_a"
        with TemporaryDirectory() as directory, patch.object(adapter_utils, "ADAPTERS_DIR", Path(directory)):
            model_path = Path(directory) / "model"
            model_path.mkdir()
            adapter_path = Path(directory) / "adapter_b"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")
            (adapter_path / "adapter_model.safetensors").write_text("model")
            with patch.object(inference, "settings", types.SimpleNamespace(model_path=str(model_path))), patch.dict(
                sys.modules, {"transformers": types.ModuleType("transformers")}
            ):
                with self.assertRaises(ImportError):
                    self.manager._load_configuration("adapter_b")
        self.assertEqual(
            self.manager.status(),
            {"loaded": False, "adapter_id": None, "generating": False},
        )

    def test_cuda_oom_unloads_model(self) -> None:
        self.manager._model = FakeModel()

        class FailingTokenizer(FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                raise Exception("oom")

        self.manager._tokenizer = FailingTokenizer()
        with patch.object(inference, "_is_cuda_oom", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                self.manager.generate("Question", None, 12, 0)
        self.assertFalse(self.manager.status()["loaded"])

    def test_generation_and_decode_errors_clear_generating(self) -> None:
        class GenerationErrorModel(FakeModel):
            def generate(self, **kwargs):
                raise ValueError("generation failed")

        self.manager._model = GenerationErrorModel()
        self.manager._tokenizer = FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "generation failed"):
            self.manager.generate("Question", None, 12, 0)
        self.assertFalse(self.manager.status()["generating"])

        class DecodeErrorTokenizer(FakeTokenizer):
            def decode(self, ids, **kwargs):
                raise ValueError("decode failed")

        self.manager._model = FakeModel()
        self.manager._tokenizer = DecodeErrorTokenizer()
        with self.assertRaisesRegex(ValueError, "decode failed"):
            self.manager.generate("Question", None, 12, 0)
        self.assertFalse(self.manager.status()["generating"])

    def test_unload_clears_all_configuration_references(self) -> None:
        self.manager._model = FakeModel()
        self.manager._tokenizer = FakeTokenizer()
        self.manager._adapter_id = "adapter"

        self.manager.unload_for_training()

        self.assertEqual(self.manager.status(), {"loaded": False, "adapter_id": None, "generating": False})


class InferenceApiTests(unittest.TestCase):
    def setUp(self) -> None:
        training._jobs.clear()
        training._gpu_owner = None
        self.client = TestClient(app)

    def tearDown(self) -> None:
        training._jobs.clear()
        training._gpu_owner = None

    def test_status_does_not_generate(self) -> None:
        response = self.client.get("/inference/status")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["loaded"])

    def test_health_reports_readiness_without_model_load(self) -> None:
        for readiness in (
            {"model_path_configured": False, "model_path_exists": False},
            {"model_path_configured": True, "model_path_exists": False},
            {"model_path_configured": True, "model_path_exists": True},
        ):
            with patch("server.model_path_readiness", return_value=readiness), patch.object(
                inference.inference_manager, "_load_configuration"
            ) as load:
                response = self.client.get("/health")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["model_path_configured"], readiness["model_path_configured"])
            self.assertEqual(response.json()["model_path_exists"], readiness["model_path_exists"])
            self.assertFalse(response.json()["model_loaded"])
            self.assertEqual(response.json()["gpu_state"], "idle")
            load.assert_not_called()

    def test_configured_frontend_origin_is_allowed_by_cors(self) -> None:
        response = self.client.get("/health", headers={"Origin": config.settings.frontend_origin})
        self.assertEqual(response.headers["access-control-allow-origin"], config.settings.frontend_origin)

    def test_model_path_readiness_distinguishes_unset_invalid_and_existing(self) -> None:
        with patch.object(config, "settings", types.SimpleNamespace(model_path="")):
            self.assertEqual(
                config.model_path_readiness(),
                {"model_path_configured": False, "model_path_exists": False},
            )
        with patch.object(config, "settings", types.SimpleNamespace(model_path="missing")):
            self.assertEqual(
                config.model_path_readiness(),
                {"model_path_configured": True, "model_path_exists": False},
            )
        with TemporaryDirectory() as directory, patch.object(
            config, "settings", types.SimpleNamespace(model_path=directory)
        ):
            self.assertEqual(
                config.model_path_readiness(),
                {"model_path_configured": True, "model_path_exists": True},
            )

    def test_startup_reconciles_evaluations(self) -> None:
        with patch("server.reconcile_evaluations") as reconcile:
            with TestClient(app):
                pass

        reconcile.assert_called_once()

    def test_request_validation_and_training_conflict(self) -> None:
        self.assertEqual(self.client.post("/inference/generate", json={"prompt": ""}).status_code, 422)
        self.assertEqual(self.client.post("/inference/generate", json={"prompt": "x", "messages": [{"role": "user", "content": "x"}]}).status_code, 422)
        self.assertEqual(self.client.post("/inference/generate", json={"messages": [{"role": "assistant", "content": "x"}]}).status_code, 422)
        self.assertEqual(self.client.post("/inference/generate", json={"prompt": "x", "max_new_tokens": 0}).status_code, 422)
        self.assertEqual(self.client.post("/inference/generate", json={"prompt": "x", "adapter_id": "../bad"}).status_code, 422)
        for job_status in ("queued", "running"):
            training._jobs["active"] = {"status": job_status}
            training._reserve_training()
            response = self.client.post("/inference/generate", json={"prompt": "x"})
            self.assertEqual(response.status_code, 409)
            training._release_training()

    def test_generation_errors_return_http_errors(self) -> None:
        with patch.object(inference.inference_manager, "generate", side_effect=FileNotFoundError("Adapter not found")):
            self.assertEqual(self.client.post("/inference/generate", json={"prompt": "x"}).status_code, 404)
        with patch.object(inference.inference_manager, "generate", side_effect=RuntimeError("Model loading failed")):
            self.assertEqual(self.client.post("/inference/generate", json={"prompt": "x"}).status_code, 500)

    def test_base_and_adapter_requests_return_generation(self) -> None:
        with patch.object(inference.inference_manager, "generate", return_value={"text": "base answer", "adapter_id": None}):
            response = self.client.post("/inference/generate", json={"prompt": "x"})
            self.assertEqual(response.json(), {"text": "base answer", "adapter_id": None, "json_valid": False, "parsed_json": None})
        with patch.object(
            inference.inference_manager,
            "generate",
            return_value={"text": "adapter answer", "adapter_id": "okved_v1"},
        ):
            response = self.client.post(
                "/inference/generate", json={"prompt": "x", "adapter_id": "okved_v1"}
            )
            self.assertEqual(response.json(), {"text": "adapter answer", "adapter_id": "okved_v1", "json_valid": False, "parsed_json": None})

    def test_messages_request_and_json_response(self) -> None:
        messages = [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "data"}]
        with patch.object(
            inference.inference_manager,
            "generate_messages",
            return_value={"text": '{"value":null}', "adapter_id": "adapter"},
        ) as generate:
            response = self.client.post(
                "/inference/generate", json={"messages": messages, "adapter_id": "adapter"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": '{"value":null}', "adapter_id": "adapter", "json_valid": True, "parsed_json": {"value": None}})
        self.assertEqual(generate.call_args.args[0], messages)
