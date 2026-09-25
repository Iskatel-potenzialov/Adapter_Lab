import io
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import dataset_utils
import train
import training
from server import app


def example(number: int) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"question {number}"},
                {"role": "assistant", "content": json.dumps({"answer": number})},
            ]
        },
        ensure_ascii=False,
    ) + "\n"


def structured_example(value: object, group_id: str | None = "group") -> str:
    row = {
        "task_profile": "structured_json",
        "evaluator": "json",
        "messages": [
            {"role": "user", "content": "source data"},
            {"role": "assistant", "content": json.dumps(value)},
        ],
    }
    if group_id is not None:
        row["group_id"] = group_id
    return json.dumps(row) + "\n"


def expert_example(group_id: str = "group", evaluation_profile: str = "diagnostic", answer: str = "reference") -> str:
    return json.dumps({"task_profile": "expert", "evaluation_profile": evaluation_profile, "group_id": group_id, "evaluator": "llm_judge", "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": answer}]}) + "\n"


class DatasetLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.datasets_dir = Path(self.temporary.name) / "datasets"
        self.datasets_dir.mkdir()
        self.patch = patch.object(dataset_utils, "DATASETS_DIR", self.datasets_dir)
        self.patch.start()
        training._jobs.clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        training._jobs.clear()
        self.patch.stop()
        self.temporary.cleanup()

    def _source(self, dataset_id: str, count: int) -> Path:
        directory = self.datasets_dir / dataset_id
        directory.mkdir()
        path = directory / "full.jsonl"
        path.write_text("".join(example(number) for number in range(count)), encoding="utf-8")
        return path

    def _lines(self, path: Path) -> list[str]:
        return path.read_text(encoding="utf-8").splitlines()

    def _group_source(self, dataset_id: str, sizes: list[int]) -> Path:
        directory = self.datasets_dir / dataset_id
        directory.mkdir()
        examples = []
        for group_number, size in enumerate(sizes):
            for example_number in range(size):
                value = json.loads(example(group_number * 100 + example_number))
                value["group_id"] = f"group-{group_number}"
                examples.append(json.dumps(value) + "\n")
        path = directory / "full.jsonl"
        path.write_text("".join(examples), encoding="utf-8")
        return path

    def _predefined(self, start: int = 0) -> dict[str, str]:
        result: dict[str, str] = {}
        for offset, name in enumerate(dataset_utils.SPLITS):
            row = json.loads(example(start + offset))
            row["group_id"] = f"{name}-{start}"
            result[name] = json.dumps(row) + "\n"
        return result

    def _upload_predefined(self, files: dict[str, str]):
        return self.client.post(
            "/api/datasets/upload-predefined",
            files={name: (f"{name}.jsonl", content, "application/json") for name, content in files.items()},
        )

    def test_canonical_split_is_deterministic_disjoint_and_complete_for_any_size(self) -> None:
        dataset_id = "a" * 32
        self._source(dataset_id, 17)
        first = dataset_utils.create_canonical_split(dataset_id)
        self.assertEqual(first, dataset_utils.create_canonical_split(dataset_id))

        split_lines = [self._lines(dataset_utils.split_path(dataset_id, split)) for split in dataset_utils.SPLITS]
        self.assertEqual(sum(map(len, split_lines)), 17)
        self.assertEqual(len(set().union(*map(set, split_lines))), 17)

        other_id = "b" * 32
        self._source(other_id, 17)
        other = dataset_utils.create_canonical_split(other_id)
        self.assertEqual(first["source_hash"], other["source_hash"])
        self.assertEqual(first["train_hash"], other["train_hash"])

    def test_group_aware_split_keeps_groups_disjoint_and_is_deterministic(self) -> None:
        dataset_id = "6" * 32
        self._group_source(dataset_id, [1, 2, 3, 5, 2, 1])
        first = dataset_utils.create_canonical_split(dataset_id)
        self.assertEqual(first["split_mode"], "group")
        self.assertEqual(first["source_group_count"], 6)
        self.assertEqual(first, dataset_utils.create_canonical_split(dataset_id))

        group_sets = []
        example_count = 0
        for split in dataset_utils.SPLITS:
            rows = [json.loads(line) for line in self._lines(dataset_utils.split_path(dataset_id, split))]
            group_sets.append({row["group_id"] for row in rows})
            example_count += len(rows)
            self.assertEqual(len(rows), first["split"][f"{split}_example_count"])
            self.assertEqual(len(group_sets[-1]), first["split"][f"{split}_group_count"])
            self.assertTrue(all("group_id" in row for row in rows))
        self.assertEqual(example_count, 14)
        self.assertTrue(all(group_sets))
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])

        other_id = "7" * 32
        self._group_source(other_id, [1, 2, 3, 5, 2, 1])
        other = dataset_utils.create_canonical_split(other_id)
        self.assertEqual(first["train_hash"], other["train_hash"])
        self.assertEqual(dataset_utils.dataset_lineage(dataset_id)["split_mode"], "group")

    def test_duplicate_content_across_groups_does_not_break_group_allocation(self) -> None:
        dataset_id = "a" * 31 + "0"
        directory = self.datasets_dir / dataset_id
        directory.mkdir()
        row = json.loads(example(1))
        rows = []
        for group_id in ("one", "two", "three"):
            for _ in range(2):
                rows.append(json.dumps({"group_id": group_id, "messages": row["messages"]}) + "\n")
        (directory / "full.jsonl").write_text("".join(rows), encoding="utf-8")
        dataset_utils.create_canonical_split(dataset_id)
        group_sets = [
            {json.loads(line)["group_id"] for line in self._lines(dataset_utils.split_path(dataset_id, split))}
            for split in dataset_utils.SPLITS
        ]
        self.assertEqual(sum(len(groups) for groups in group_sets), 3)
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])

    def test_group_validation_mixed_and_small_group_datasets_are_rejected(self) -> None:
        dataset_id = "0" * 32
        directory = self.datasets_dir / dataset_id
        directory.mkdir()
        grouped = json.loads(example(1))
        grouped["group_id"] = "one"
        ungrouped = json.loads(example(2))
        (directory / "full.jsonl").write_text(
            json.dumps(grouped) + "\n" + json.dumps(ungrouped) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Mixed group_id"):
            dataset_utils.create_canonical_split(dataset_id)

        for dataset_id, value in (("4" * 32, ""), ("5" * 32, 3)):
            path = self.datasets_dir / f"{dataset_id}.jsonl"
            row = json.loads(example(1))
            row["group_id"] = value
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            validation = dataset_utils.validate_jsonl(path)
            self.assertFalse(validation["valid"])
            self.assertIn("group_id", validation["errors"][0]["message"])

        self._group_source("3" * 32, [10, 10])
        with self.assertRaisesRegex(ValueError, "3 unique"):
            dataset_utils.create_canonical_split("3" * 32)

    def test_optional_evaluator_validation_and_hashes(self) -> None:
        rows = []
        for evaluator in (None, "exact", "normalized_text", "json"):
            row = json.loads(example(len(rows)))
            if evaluator:
                row["evaluator"] = evaluator
            rows.append(json.dumps(row) + "\n")
        path = self.datasets_dir / f"{'b' * 32}.jsonl"
        path.write_text("".join(rows), encoding="utf-8")
        self.assertTrue(dataset_utils.validate_jsonl(path)["valid"])
        for dataset_id, evaluator in (("c" * 32, "fuzzy"), ("d" * 32, ""), ("e" * 32, 7)):
            invalid = json.loads(example(1))
            invalid["evaluator"] = evaluator
            invalid_path = self.datasets_dir / f"{dataset_id}.jsonl"
            invalid_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            validation = dataset_utils.validate_jsonl(invalid_path)
            self.assertFalse(validation["valid"])
            self.assertIn("evaluator", validation["errors"][0]["message"])

        self._source("f" * 32, 3)
        changed = json.loads(example(0))
        changed["evaluator"] = "exact"
        (self.datasets_dir / ("f" * 32) / "full.jsonl").write_text(
            json.dumps(changed) + "\n" + "".join(example(number) for number in range(1, 3)), encoding="utf-8"
        )
        self.assertNotEqual(dataset_utils.content_hash(path), dataset_utils.content_hash(dataset_utils.dataset_path("f" * 32)))

    def test_expert_profile_validation_metadata_and_group_split(self) -> None:
        path = self.datasets_dir / "expert.jsonl"
        path.write_text("".join(expert_example(f"group-{number}") for number in range(3)), encoding="utf-8")
        validation = dataset_utils.validate_jsonl(path)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["profile"], {"task_profile": "expert", "evaluation_profile": "diagnostic", "evaluator": "llm_judge", "n_examples": 3, "n_groups": 3})
        dataset_id = "9" * 32
        directory = self.datasets_dir / dataset_id
        directory.mkdir()
        (directory / "full.jsonl").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        self.assertEqual(dataset_utils.create_canonical_split(dataset_id)["split_mode"], "group")

    def test_expert_profile_rejects_invalid_or_mixed_metadata(self) -> None:
        cases = [
            expert_example().replace('"llm_judge"', '"exact"'),
            expert_example().replace('"group_id": "group", ', ""),
            expert_example().replace('"evaluation_profile": "diagnostic", ', ""),
            expert_example(evaluation_profile="other"),
            expert_example(answer="   "),
            expert_example("one") + expert_example("two", "other"),
            expert_example("one") + structured_example({"a": 1}, "two"),
        ]
        for index, content in enumerate(cases):
            path = self.datasets_dir / f"invalid-{index}.jsonl"
            path.write_text(content, encoding="utf-8")
            self.assertFalse(dataset_utils.validate_jsonl(path)["valid"], content)

    def test_evaluator_is_preserved_and_ignored_by_group_allocation_and_training_text(self) -> None:
        dataset_id = "b" * 31 + "0"
        directory = self.datasets_dir / dataset_id
        directory.mkdir()
        rows = []
        for group_id, evaluators in (("shared", ["exact", "normalized_text", "json"]), ("second", ["exact"]), ("third", ["json"])):
            for evaluator in evaluators:
                row = json.loads(example(len(rows)))
                row["group_id"] = group_id
                row["evaluator"] = evaluator
                rows.append(row)
        (directory / "full.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        dataset_utils.create_canonical_split(dataset_id)
        split_rows = [
            json.loads(line)
            for split in dataset_utils.SPLITS
            for line in self._lines(dataset_utils.split_path(dataset_id, split))
        ]
        self.assertEqual({row["evaluator"] for row in split_rows}, {"exact", "normalized_text", "json"})
        shared_splits = [
            split for split in dataset_utils.SPLITS
            if any(json.loads(line)["group_id"] == "shared" for line in self._lines(dataset_utils.split_path(dataset_id, split)))
        ]
        self.assertEqual(len(shared_splits), 1)

        row = split_rows[0]
        received: dict = {}
        def apply_template(messages: list[dict], **_: object) -> str:
            received["messages"] = messages
            return "formatted"

        text = train.format_messages(
            row["messages"],
            apply_template,
        )
        self.assertEqual(text, "formatted")
        self.assertEqual(received["messages"], row["messages"])
        self.assertTrue(all("group_id" not in message and "evaluator" not in message for message in received["messages"]))

    def test_upload_preview_and_split_preserve_partial_evaluator_metadata(self) -> None:
        rows = []
        for evaluator in ("exact", None, "json"):
            row = json.loads(example(len(rows)))
            if evaluator:
                row["evaluator"] = evaluator
            rows.append(json.dumps(row) + "\n")
        response = self.client.post(
            "/api/datasets/upload", files={"file": ("evaluated.jsonl", "".join(rows), "application/json")}
        )
        self.assertEqual(response.status_code, 201)
        dataset_id = response.json()["id"]
        preview = self.client.get(f"/api/datasets/{dataset_id}/preview").json()["examples"]
        self.assertEqual(preview[0]["evaluator"], "exact")
        self.assertNotIn("evaluator", preview[1])
        self.assertEqual(self.client.post(f"/api/datasets/{dataset_id}/split", json={}).status_code, 200)
        saved = [
            json.loads(line)
            for split in dataset_utils.SPLITS
            for line in self._lines(dataset_utils.split_path(dataset_id, split))
        ]
        self.assertEqual(sum("evaluator" in row for row in saved), 2)

    def test_predefined_upload_preserves_splits_and_rejects_canonical_split(self) -> None:
        files = self._predefined()
        response = self._upload_predefined(files)
        self.assertEqual(response.status_code, 201)
        value = response.json()
        dataset_id = value["id"]
        directory = self.datasets_dir / dataset_id
        self.assertEqual(value["dataset"]["dataset_mode"], "predefined")
        self.assertNotIn("split_mode", value["dataset"])
        self.assertEqual(value["dataset"]["split"]["seed"], None)
        for name in dataset_utils.SPLITS:
            path = directory / f"{name}.jsonl"
            self.assertEqual(path.read_text(encoding="utf-8"), files[name])
            self.assertEqual(value["splits"][name]["n_examples"], 1)
            self.assertEqual(value["splits"][name]["n_groups"], 1)
            self.assertEqual(value["dataset"][f"{name}_hash"], dataset_utils.content_hash(path))
        self.assertEqual((directory / "full.jsonl").read_text(encoding="utf-8"), "".join(files[name] for name in dataset_utils.SPLITS))
        self.assertEqual(self.client.post(f"/api/datasets/{dataset_id}/split", json={}).status_code, 409)
        listed = self.client.get("/api/datasets").json()[0]
        self.assertEqual(listed["dataset_mode"], "predefined")
        self.assertEqual(dataset_utils.dataset_lineage(dataset_id)["split_seed"], None)

        preview = self.client.get(f"/api/datasets/{dataset_id}/preview")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(len(preview.json()["examples"]), 3)

        self.assertEqual(self.client.delete(f"/api/datasets/{dataset_id}").status_code, 204)
        self.assertFalse(directory.exists())

    def test_predefined_requires_disjoint_groups_and_valid_files(self) -> None:
        for invalid_name in dataset_utils.SPLITS:
            files = self._predefined()
            files[invalid_name] = "not json\n"
            with self.subTest(invalid=invalid_name):
                self.assertEqual(self._upload_predefined(files).status_code, 422)
                self.assertEqual(list(self.datasets_dir.iterdir()), [])

        missing_group = self._predefined()
        missing_group["train"] = example(1)
        self.assertEqual(self._upload_predefined(missing_group).status_code, 422)
        self.assertEqual(list(self.datasets_dir.iterdir()), [])

        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            files = self._predefined()
            shared = "shared"
            for name in (left, right):
                row = json.loads(files[name]); row["group_id"] = shared; files[name] = json.dumps(row) + "\n"
            with self.subTest(overlap=f"{left}/{right}"):
                self.assertEqual(self._upload_predefined(files).status_code, 422)
                self.assertEqual(list(self.datasets_dir.iterdir()), [])

        repeated = self._predefined()
        train_row = json.loads(repeated["train"])
        repeated["train"] += json.dumps(train_row) + "\n"
        self.assertEqual(self._upload_predefined(repeated).status_code, 201)

    def test_predefined_profile_consistency_and_legacy_metadata(self) -> None:
        expert = {name: expert_example(f"{name}-group") for name in dataset_utils.SPLITS}
        expert_response = self._upload_predefined(expert)
        self.assertEqual(expert_response.status_code, 201)
        self.assertEqual(expert_response.json()["validation"]["profile"]["task_profile"], "expert")

        structured = {
            "train": structured_example({"a": 1}, "train-group"),
            "validation": structured_example({"a": 2}, "validation-group"),
            "test": structured_example({"b": 3}, "test-group"),
        }
        self.assertEqual(self._upload_predefined(structured).status_code, 422)

        incompatible = self._predefined()
        incompatible["train"] = expert_example("expert-group")
        self.assertEqual(self._upload_predefined(incompatible).status_code, 422)

        dataset_id = "c" * 32
        self._source(dataset_id, 3)
        dataset_utils.create_canonical_split(dataset_id)
        metadata_path = self.datasets_dir / dataset_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.pop("dataset_mode")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertEqual(dataset_utils.dataset_lineage(dataset_id)["dataset_mode"], "canonical")

    def test_small_and_invalid_ratio_datasets_are_rejected(self) -> None:
        self._source("c" * 32, 2)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            dataset_utils.create_canonical_split("c" * 32)
        self._source("d" * 32, 3)
        split = dataset_utils.create_canonical_split("d" * 32)
        self.assertEqual([split["split"][f"{name}_count"] for name in dataset_utils.SPLITS], [1, 1, 1])
        self._source("e" * 32, 10)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            dataset_utils.create_canonical_split("e" * 32, 0.8, 0.1, 0.2)

    def test_duplicate_examples_are_preserved(self) -> None:
        dataset_id = "9" * 32
        directory = self.datasets_dir / dataset_id
        directory.mkdir()
        (directory / "full.jsonl").write_text(example(1) * 10, encoding="utf-8")
        dataset_utils.create_canonical_split(dataset_id)
        self.assertEqual(
            sum(len(self._lines(dataset_utils.split_path(dataset_id, split))) for split in dataset_utils.SPLITS),
            10,
        )

    def test_concurrent_same_split_requests_share_one_canonical_result(self) -> None:
        dataset_id = "8" * 32
        self._source(dataset_id, 100)
        results: list[dict] = []
        errors: list[Exception] = []

        def split() -> None:
            try:
                results.append(dataset_utils.create_canonical_split(dataset_id))
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=split) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_api_split_status_conflict_and_safe_delete(self) -> None:
        dataset_id = "f" * 32
        self._source(dataset_id, 10)
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["split_status"], "not_created")

        response = self.client.post(f"/api/datasets/{dataset_id}/split", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["split"]["train_count"], 8)
        response = self.client.post(f"/api/datasets/{dataset_id}/split", json={"seed": 7})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.get("/api/datasets").json()[0]["split_status"], "ready")

        training._jobs["active"] = {"status": "running", "dataset": dataset_id}
        self.assertEqual(self.client.delete(f"/api/datasets/{dataset_id}").status_code, 409)
        training._jobs.clear()
        self.assertEqual(self.client.delete(f"/api/datasets/{dataset_id}").status_code, 204)
        self.assertFalse((self.datasets_dir / dataset_id).exists())

    def test_legacy_dataset_is_readable_without_migration_and_can_be_split(self) -> None:
        dataset_id = "1" * 32
        legacy = self.datasets_dir / f"{dataset_id}.jsonl"
        legacy.write_text("".join(example(number) for number in range(10)), encoding="utf-8")
        self.assertEqual(dataset_utils.list_datasets()[0]["split_status"], "not_created")
        self.assertFalse((self.datasets_dir / dataset_id).exists())
        dataset_utils.create_canonical_split(dataset_id)
        self.assertEqual(dataset_utils.training_dataset_path(dataset_id).name, "train.jsonl")
        dataset_utils.delete_dataset(dataset_id)
        self.assertFalse(legacy.exists())
        self.assertFalse((self.datasets_dir / dataset_id).exists())

    def test_upload_preview_and_path_traversal(self) -> None:
        response = self.client.post(
            "/api/datasets/upload",
            files={"file": ("sample.jsonl", "".join(example(number) for number in range(5)), "application/json")},
        )
        self.assertEqual(response.status_code, 201)
        dataset_id = response.json()["id"]
        self.assertEqual(response.json()["validation"]["n_examples"], 5)
        self.assertEqual(self.client.get(f"/api/datasets/{dataset_id}/preview?n=2").status_code, 200)
        self.assertIn(self.client.get("/api/datasets/..%2Fescape/preview").status_code, {400, 404})

    def test_structured_json_profile_validation_and_metadata(self) -> None:
        empty = self.datasets_dir / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        self.assertFalse(dataset_utils.validate_jsonl(empty)["valid"])
        self.assertEqual(
            self.client.post(
                "/api/datasets/upload", files={"file": ("empty.jsonl", "", "application/json")}
            ).status_code,
            422,
        )

        for value, message in (("not json", "invalid JSON"), ([], "top-level JSON object")):
            path = self.datasets_dir / "invalid.jsonl"
            content = structured_example({"value": 1}) if value == "not json" else structured_example(value)
            if value == "not json":
                row = json.loads(content)
                row["messages"][-1]["content"] = value
                content = json.dumps(row) + "\n"
            path.write_text(content, encoding="utf-8")
            self.assertIn(message, dataset_utils.validate_jsonl(path)["errors"][0]["message"])

        missing_group = self.datasets_dir / "missing-group.jsonl"
        missing_group.write_text(structured_example({"value": 1}, None), encoding="utf-8")
        self.assertIn("group_id", dataset_utils.validate_jsonl(missing_group)["errors"][0]["message"])

        inconsistent = self.datasets_dir / "inconsistent.jsonl"
        inconsistent.write_text(
            structured_example({"a": 1}, "one") + structured_example({"b": 1}, "two"),
            encoding="utf-8",
        )
        self.assertIn("top-level keys", dataset_utils.validate_jsonl(inconsistent)["errors"][0]["message"])

        content = (
            structured_example({"name": "first", "enabled": True, "section": None}, "one")
            + structured_example({"name": "second", "enabled": False, "section": "A"}, "two")
            + structured_example({"name": "third", "enabled": True, "section": "B"}, "three")
        )
        response = self.client.post(
            "/api/datasets/upload", files={"file": ("structured.jsonl", content, "application/json")}
        )
        self.assertEqual(response.status_code, 201)
        profile = response.json()["validation"]["profile"]
        self.assertEqual(profile["target_schema"]["fields"], ["enabled", "name", "section"])
        self.assertEqual(profile["target_schema"]["field_stats"]["enabled"]["observed_types"], ["boolean"])
        self.assertEqual(profile["target_schema"]["field_stats"]["section"]["null_count"], 1)
        self.assertEqual(profile["null_statistics"], {"total_null_values": 1, "examples_with_null": 1, "examples_without_null": 2})
        dataset_id = response.json()["id"]
        self.assertEqual(
            json.loads((self.datasets_dir / dataset_id / "metadata.json").read_text())["profile"]["target_schema"]["fields"],
            ["enabled", "name", "section"],
        )
        split = self.client.post(f"/api/datasets/{dataset_id}/split", json={})
        self.assertEqual(split.status_code, 200)
        self.assertEqual(split.json()["profile"]["task_profile"], "structured_json")
        self.assertEqual(dataset_utils.dataset_lineage(dataset_id)["profile"]["task_profile"], "structured_json")

    def test_legacy_json_evaluator_remains_generic(self) -> None:
        path = self.datasets_dir / "legacy.jsonl"
        row = json.loads(example(1))
        row["evaluator"] = "json"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        validation = dataset_utils.validate_jsonl(path)
        self.assertTrue(validation["valid"])
        self.assertNotIn("profile", validation)

    def test_training_requires_and_uses_only_canonical_train_split(self) -> None:
        dataset_id = "2" * 32
        self._source(dataset_id, 10)
        self.assertEqual(self.client.post("/training/start", json={"dataset_id": dataset_id}).status_code, 409)
        dataset_utils.create_canonical_split(dataset_id)

        class Process:
            pid = 1
            stdout = io.StringIO('{"event":"done"}\n')

            def wait(self) -> int:
                return 0

        adapters = self.datasets_dir.parent / "adapters"
        experiments = self.datasets_dir.parent / "experiments"
        logs = self.datasets_dir.parent / "logs"
        with patch.object(training, "ADAPTERS_DIR", adapters), patch.object(
            training, "EXPERIMENTS_DIR", experiments
        ), patch.object(training, "LOGS_DIR", logs), patch.object(
            training.subprocess, "Popen", return_value=Process()
        ) as popen:
            training.start_training(dataset_id, "split_job", {})
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--dataset") + 1], str(dataset_utils.training_dataset_path(dataset_id)))
            self.assertEqual(command[command.index("--validation") + 1], str(dataset_utils.validation_dataset_path(dataset_id)))
            self.assertNotIn("test.jsonl", command)
