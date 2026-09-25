import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import training
import inference
import adapter_utils
from server import app


class FakeProcess:
    def __init__(self, lines: list[str], return_code: int = 0):
        self.pid = 1234
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self.return_code = return_code

    def wait(self) -> int:
        return self.return_code


class TrainingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.dataset_dir = self.root / "datasets"
        self.adapters_dir = self.root / "adapters"
        self.experiments_dir = self.root / "experiments"
        self.logs_dir = self.root / "logs"
        self.dataset_dir.mkdir()
        (self.dataset_dir / f"{'a' * 32}.jsonl").write_text("{}", encoding="utf-8")
        training._jobs.clear()
        training._gpu_owner = None
        def fake_training_dataset_path(dataset_id: str) -> Path:
            predefined = self.dataset_dir / dataset_id / "train.jsonl"
            if predefined.is_file():
                return predefined
            path = self.dataset_dir / f"{dataset_id}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(dataset_id)
            return path

        def fake_validation_dataset_path(dataset_id: str) -> Path:
            predefined = self.dataset_dir / dataset_id / "validation.jsonl"
            if predefined.is_file():
                return predefined
            fake_training_dataset_path(dataset_id)
            return self.dataset_dir / f"{dataset_id}_validation.jsonl"

        def fake_dataset_lineage(dataset_id: str) -> dict:
            fake_training_dataset_path(dataset_id)
            return {
                "training_split": "train",
                "validation_split": "validation",
                "source_hash": "source",
                "train_hash": "train",
                "validation_hash": "validation",
                "validation_count": 1,
                "split_seed": 42,
                "split_config": {},
            }

        self.patches = [
            patch.object(training, "ADAPTERS_DIR", self.adapters_dir),
            patch.object(training, "EXPERIMENTS_DIR", self.experiments_dir),
            patch.object(training, "LOGS_DIR", self.logs_dir),
            patch.object(training, "training_dataset_path", fake_training_dataset_path),
            patch.object(training, "validation_dataset_path", fake_validation_dataset_path),
            patch.object(training, "dataset_lineage", fake_dataset_lineage),
            patch.object(adapter_utils, "ADAPTERS_DIR", self.adapters_dir),
        ]
        for item in self.patches:
            item.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        training._gpu_owner = None
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def test_done_event_completes_job(self) -> None:
        lines = [
            json.dumps({"event": "started", "exp_id": "job1"}),
            json.dumps({"event": "metrics", "step": 2, "loss": 1.2, "lr": 0.001, "epoch": 0.5}),
            json.dumps({"event": "metrics", "step": 2, "eval_loss": 0.8, "eval_runtime": 1.5, "epoch": 1.0}),
            json.dumps({"event": "done", "exp_id": "job1", "best_epoch": 1.0, "best_eval_loss": 0.8, "best_checkpoint": "adapter/checkpoint-2"}),
        ]
        with patch.object(training.subprocess, "Popen", return_value=FakeProcess(lines)) as popen:
            response = self.client.post("/training/start", json={"dataset_id": "a" * 32, "job_id": "job1"})
            self.assertEqual(response.status_code, 201)
            command = popen.call_args.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertIn("--dataset", command)
            self.assertIn("--validation", command)
            self.assertIn("--output", command)
            self.assertIn("--config", command)
            self.assertEqual(command[command.index("--validation") + 1], str(self.dataset_dir / f"{'a' * 32}_validation.jsonl"))
            self.assertNotIn("test.jsonl", command)
            self.assertEqual(popen.call_args.kwargs["cwd"], str(training.BACKEND_DIR))
            self.assertFalse(popen.call_args.kwargs.get("shell", False))
            self.assertEqual(popen.call_args.kwargs["stderr"], training.subprocess.STDOUT)
            self.assertEqual(
                json.loads((self.experiments_dir / "job1_config.json").read_text(encoding="utf-8")),
                {
                    "exp_id": "job1",
                    "epochs": 3,
                    "batch_size": 1,
                    "eval_batch_size": 1,
                    "grad_accum": 16,
                    "learning_rate": 0.0002,
                    "lr_scheduler_type": "cosine",
                    "warmup_ratio": 0.03,
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "max_seq_length": 256,
                    "early_stopping_patience": None,
                    "early_stopping_threshold": 0,
                },
            )
            training._jobs["job1"]["_process"]
            training._jobs["job1"]["status"]
            import time

            for _ in range(20):
                job = self.client.get("/training/job1").json()
                if job["finished_at"]:
                    break
                time.sleep(0.01)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["current_step"], 2)
            self.assertEqual(job["loss"], 1.2)
            self.assertEqual(job["eval_loss"], 0.8)
            self.assertEqual(job["eval_runtime"], 1.5)
            self.assertEqual(job["best_epoch"], 1.0)
            self.assertEqual(job["best_eval_loss"], 0.8)
            self.assertEqual(job["best_checkpoint"], "adapter/checkpoint-2")
            self.assertEqual(job["config"]["eval_batch_size"], 1)
            self.assertIsNone(job["config"]["early_stopping_patience"])
            self.assertEqual(job["adapter_path"], job["output"])
            self.assertIsNone(job["progress"])
            self.assertEqual(len(self.client.get("/training").json()), 1)
            self.assertIn('"event": "done"', Path(job["log_path"]).read_text(encoding="utf-8"))

    def test_failed_process_and_missing_job(self) -> None:
        with patch.object(training.subprocess, "Popen", return_value=FakeProcess(["unexpected output"], 1)):
            response = self.client.post("/training/start", json={"dataset_id": "a" * 32, "job_id": "job2"})
            self.assertEqual(response.status_code, 201)
            import time

            for _ in range(20):
                job = self.client.get("/training/job2").json()
                if job["finished_at"]:
                    break
                time.sleep(0.01)
            self.assertEqual(job["status"], "failed")
        self.assertEqual(self.client.get("/training/missing").status_code, 404)

    def test_zero_exit_without_done_fails_job(self) -> None:
        with patch.object(training.subprocess, "Popen", return_value=FakeProcess([json.dumps({"event": "started"})], 0)):
            response = self.client.post("/training/start", json={"dataset_id": "a" * 32, "job_id": "job3"})
            self.assertEqual(response.status_code, 201)
            import time

            for _ in range(20):
                job = self.client.get("/training/job3").json()
                if job["finished_at"]:
                    break
                time.sleep(0.01)
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["error"], "Training process exited without a done event")

    def test_done_event_does_not_release_running_job_before_exit(self) -> None:
        training._jobs["job5"] = {"status": "running", "_done": False, "adapter_path": None, "output": "adapter", "events": []}
        training._handle_output("job5", json.dumps({"event": "done"}))
        self.assertEqual(training._jobs["job5"]["status"], "running")
        self.assertTrue(training._jobs["job5"]["_done"])

    def test_model_path_error_is_recorded(self) -> None:
        lines = [json.dumps({"event": "error", "message": "MODEL_PATH must point to a local model directory"})]
        with patch.object(training.subprocess, "Popen", return_value=FakeProcess(lines, 1)):
            response = self.client.post("/training/start", json={"dataset_id": "a" * 32, "job_id": "job4"})
            self.assertEqual(response.status_code, 201)
            import time

            for _ in range(20):
                job = self.client.get("/training/job4").json()
                if job["finished_at"]:
                    break
                time.sleep(0.01)
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["error"], "MODEL_PATH must point to a local model directory")

    def test_rejects_missing_dataset_and_concurrent_job(self) -> None:
        self.assertEqual(self.client.post("/training/start", json={"dataset_id": "b" * 32}).status_code, 404)
        self.assertEqual(
            self.client.post("/training/start", json={"dataset_id": "a" * 32, "job_id": "../escape"}).status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/training/start", json={"dataset_id": "a" * 32, "target_modules": []}).status_code,
            422,
        )
        with patch.object(training.subprocess, "Popen", return_value=FakeProcess([], 0)):
            training._jobs["running"] = {"status": "running"}
            response = self.client.post("/training/start", json={"dataset_id": "a" * 32})
            self.assertEqual(response.status_code, 409)

    def test_training_is_rejected_while_inference_owns_gpu(self) -> None:
        inference.inference_manager._model = object()
        training.reserve_inference()
        try:
            response = self.client.post("/training/start", json={"dataset_id": "a" * 32})
            self.assertEqual(response.status_code, 409)
            self.assertIsNotNone(inference.inference_manager._model)
        finally:
            training.release_inference()
            inference.inference_manager._model = None

    def test_training_releases_idle_inference_model_before_start(self) -> None:
        lines = [json.dumps({"event": "done"})]
        with patch.object(inference.inference_manager, "unload_for_training") as unload, patch.object(
            training.subprocess, "Popen", return_value=FakeProcess(lines)
        ):
            response = self.client.post("/training/start", json={"dataset_id": "a" * 32, "job_id": "job5"})
            self.assertEqual(response.status_code, 201)
            unload.assert_called_once()

    def test_predefined_dataset_uses_only_train_and_validation_paths(self) -> None:
        dataset_id = "p" * 32
        directory = self.dataset_dir / dataset_id
        directory.mkdir()
        for name in ("train", "validation", "test"):
            (directory / f"{name}.jsonl").write_text(name, encoding="utf-8")
        with patch.object(training.subprocess, "Popen", return_value=FakeProcess([json.dumps({"event": "done"})])) as popen:
            response = self.client.post("/training/start", json={"dataset_id": dataset_id, "job_id": "predefined"})
            self.assertEqual(response.status_code, 201)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--dataset") + 1], str(directory / "train.jsonl"))
        self.assertEqual(command[command.index("--validation") + 1], str(directory / "validation.jsonl"))
        self.assertNotIn(str(directory / "test.jsonl"), command)

    def test_metadata_is_written_only_after_successful_training(self) -> None:
        class AdapterProcess(FakeProcess):
            def __init__(self, output: Path, return_code: int):
                super().__init__([json.dumps({"event": "done"})], return_code)
                self.output = output

            def wait(self) -> int:
                self.output.parent.mkdir(exist_ok=True)
                self.output.mkdir()
                (self.output / "adapter_config.json").write_text("{}")
                (self.output / "adapter_model.safetensors").write_text("weights")
                return super().wait()

        output = self.adapters_dir / "metadata_job"
        with patch.object(training.subprocess, "Popen", return_value=AdapterProcess(output, 0)):
            response = self.client.post(
                "/training/start", json={"dataset_id": "a" * 32, "job_id": "metadata_job", "epochs": 7}
            )
            self.assertEqual(response.status_code, 201)
            import time

            for _ in range(20):
                job = self.client.get("/training/metadata_job").json()
                if job["finished_at"]:
                    break
                time.sleep(0.01)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["adapter_id"], "metadata_job")
        self.assertEqual(metadata["training_job_id"], "metadata_job")
        self.assertEqual(metadata["dataset_id"], "a" * 32)
        self.assertEqual(metadata["dataset_lineage"]["training_split"], "train")
        self.assertEqual(metadata["dataset_lineage"]["validation_split"], "validation")
        self.assertEqual(metadata["training_config"]["epochs"], 7)
        self.assertIn("best_epoch", metadata)
        self.assertIn("best_eval_loss", metadata)
        self.assertIn("best_checkpoint", metadata)

        failed_output = self.adapters_dir / "failed_metadata_job"
        with patch.object(training.subprocess, "Popen", return_value=AdapterProcess(failed_output, 1)):
            response = self.client.post(
                "/training/start", json={"dataset_id": "a" * 32, "job_id": "failed_metadata_job"}
            )
            self.assertEqual(response.status_code, 201)
            import time

            for _ in range(20):
                job = self.client.get("/training/failed_metadata_job").json()
                if job["finished_at"]:
                    break
                time.sleep(0.01)
        self.assertEqual(job["status"], "failed")
        self.assertFalse((failed_output / "metadata.json").exists())

        metadata_error_output = self.adapters_dir / "metadata_error_job"
        with patch.object(adapter_utils, "write_adapter_metadata", side_effect=OSError("disk full")), patch.object(
            training.subprocess, "Popen", return_value=AdapterProcess(metadata_error_output, 0)
        ):
            response = self.client.post(
                "/training/start", json={"dataset_id": "a" * 32, "job_id": "metadata_error_job"}
            )
            self.assertEqual(response.status_code, 201)
            import time

            for _ in range(20):
                job = self.client.get("/training/metadata_error_job").json()
                if job["finished_at"]:
                    break
                time.sleep(0.01)
        self.assertEqual(job["status"], "completed")
        self.assertIn("Could not write adapter metadata", job["metadata_error"])
        self.assertFalse((metadata_error_output / "metadata.json").exists())
