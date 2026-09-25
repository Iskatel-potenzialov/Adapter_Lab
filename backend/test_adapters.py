import os
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import adapter_utils
import inference
import training
from server import app


class AdapterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.adapters_dir = Path(self.directory.name) / "adapters"
        self.adapters_dir.mkdir()
        self.patches = [patch.object(adapter_utils, "ADAPTERS_DIR", self.adapters_dir)]
        for item in self.patches:
            item.start()
        training._jobs.clear()
        inference.inference_manager._model = None
        inference.inference_manager._adapter_id = None
        self.client = TestClient(app)

    def tearDown(self) -> None:
        training._jobs.clear()
        inference.inference_manager._model = None
        inference.inference_manager._adapter_id = None
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def _adapter(self, adapter_id: str, metadata: str | None = None) -> Path:
        path = self.adapters_dir / adapter_id
        path.mkdir()
        (path / "adapter_config.json").write_text("config")
        (path / "adapter_model.safetensors").write_text("weights")
        if metadata is not None:
            (path / "metadata.json").write_text(metadata, encoding="utf-8")
        return path

    def test_empty_list_and_deterministic_valid_invalid_order(self) -> None:
        self.assertEqual(self.client.get("/adapters").json(), [])
        self._adapter("zeta")
        invalid = self.adapters_dir / "alpha"
        invalid.mkdir()
        response = self.client.get("/adapters")
        self.assertEqual([item["adapter_id"] for item in response.json()], ["alpha", "zeta"])
        self.assertFalse(response.json()[0]["valid"])
        self.assertTrue(response.json()[1]["valid"])

    def test_detail_size_files_and_old_adapter_without_metadata(self) -> None:
        self._adapter("old_adapter")
        response = self.client.get("/adapters/old_adapter")
        detail = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(detail["valid"])
        self.assertFalse(detail["metadata_available"])
        self.assertEqual(detail["created_at_source"], "filesystem")
        self.assertEqual(detail["size_bytes"], len("config") + len("weights"))
        self.assertEqual(detail["files"], ["adapter_config.json", "adapter_model.safetensors"])
        self.assertTrue(all(not Path(name).is_absolute() for name in detail["files"]))

    def test_detail_returns_actual_metadata_without_defaults(self) -> None:
        metadata = (
            '{"adapter_id":"custom","training_job_id":"custom","dataset_id":"dataset1",'
            '"created_at":"2026-01-01T00:00:00+00:00",'
            '"dataset_lineage":{"split_mode":"group_aware","test_count":2},'
            '"training_config":{"epochs":7,"lora_r":4,"target_modules":["q_proj"]},'
            '"best_epoch":3,"best_eval_loss":0.1,"best_checkpoint":"checkpoint-12"}'
        )
        self._adapter("custom", metadata)
        detail = self.client.get("/adapters/custom").json()
        self.assertTrue(detail["metadata_available"])
        self.assertEqual(detail["created_at_source"], "metadata")
        self.assertEqual(detail["dataset_id"], "dataset1")
        self.assertEqual(detail["dataset_lineage"], {"split_mode": "group_aware", "test_count": 2})
        self.assertEqual(detail["training_config"], {"epochs": 7, "lora_r": 4, "target_modules": ["q_proj"]})
        self.assertEqual(detail["best_epoch"], 3)
        self.assertEqual(detail["best_eval_loss"], 0.1)
        self.assertEqual(detail["best_checkpoint"], "checkpoint-12")

    def test_missing_unsafe_and_invalid_adapters(self) -> None:
        self.assertEqual(self.client.get("/adapters/missing").status_code, 404)
        self.assertEqual(self.client.delete("/adapters/missing").status_code, 404)
        self.assertIn(self.client.get("/adapters/..%2Fescape").status_code, {400, 404})
        invalid = self.adapters_dir / "invalid"
        invalid.mkdir()
        detail = self.client.get("/adapters/invalid").json()
        self.assertFalse(detail["valid"])
        self.assertEqual(detail["validation_error"], "Missing adapter_config.json")

    def test_download_contains_only_deployable_root_files(self) -> None:
        path = self._adapter("downloadable")
        (path / "metadata.json").write_text("{}")
        (path / "checkpoint-1").mkdir()
        (path / "checkpoint-1" / "optimizer.pt").write_text("ignored")
        response = self.client.get("/adapters/downloadable/download")
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            self.assertEqual(sorted(archive.namelist()), ["adapter_config.json", "adapter_model.safetensors", "metadata.json"])
        self.assertEqual(self.client.get("/adapters/missing/download").status_code, 404)
        self.assertIn(self.client.get("/adapters/..%2Fescape/download").status_code, {400, 404})

    def test_delete_unused_loaded_and_training_output_adapters(self) -> None:
        path = self._adapter("unused")
        self.assertEqual(self.client.delete("/adapters/unused").status_code, 204)
        self.assertFalse(path.exists())

        self._adapter("loaded")
        inference.inference_manager._model = object()
        inference.inference_manager._adapter_id = "loaded"
        self.assertEqual(self.client.delete("/adapters/loaded").status_code, 409)
        inference.inference_manager._model = None
        inference.inference_manager._adapter_id = None

        self._adapter("generating")
        inference.inference_manager._model = object()
        inference.inference_manager._adapter_id = "generating"
        inference.inference_manager._generating = True
        self.assertEqual(self.client.delete("/adapters/generating").status_code, 409)
        inference.inference_manager._model = None
        inference.inference_manager._adapter_id = None
        inference.inference_manager._generating = False

        self._adapter("training_output")
        training._jobs["job"] = {"status": "running", "output": str(self.adapters_dir / "training_output")}
        self.assertEqual(self.client.delete("/adapters/training_output").status_code, 409)

    def test_symlink_escape_is_not_listed_or_deleted(self) -> None:
        outside = Path(self.directory.name) / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep")
        escaped = self.adapters_dir / "escaped"
        try:
            os.symlink(outside, escaped, target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation is unavailable on this platform")
        self.assertEqual(self.client.get("/adapters").json(), [])
        self.assertEqual(self.client.delete("/adapters/escaped").status_code, 400)
        self.assertTrue((outside / "keep.txt").exists())
