import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datasets import Dataset

import train


class PromptCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = [
            {"role": "system", "content": "Ты — справочник ОКВЭД-2."},
            {"role": "user", "content": "Верни полную запись."},
            {"role": "assistant", "content": '{"code":"15.20.5"}'},
        ]

    def test_target_is_completion_not_prompt(self) -> None:
        value = train.prompt_completion(self.messages)
        self.assertEqual(value["prompt"], self.messages[:-1])
        self.assertEqual(value["completion"], [self.messages[-1]])
        self.assertNotIn(self.messages[-1], value["prompt"])
        self.assertNotIn("text", value)

    def test_dataset_metadata_is_not_in_training_representation(self) -> None:
        example = {
            "group_id": "example-group",
            "evaluator": "json",
            "messages": self.messages,
        }

        value = train.prompt_completion(example["messages"])

        self.assertEqual(set(value), {"prompt", "completion"})
        self.assertNotIn("group_id", value)
        self.assertNotIn("evaluator", value)

    def test_train_and_validation_share_preprocessing(self) -> None:
        train_dataset = Dataset.from_list(
            [{"group_id": "train", "messages": self.messages}]
        )
        validation_dataset = Dataset.from_list(
            [{"evaluator": "json", "messages": self.messages}]
        )

        prepared_train = train.prepare_dataset(train_dataset)
        prepared_validation = train.prepare_dataset(validation_dataset)

        self.assertEqual(prepared_train.column_names, ["prompt", "completion"])
        self.assertEqual(prepared_validation.column_names, ["prompt", "completion"])
        self.assertEqual(prepared_train[0]["prompt"], self.messages[:-1])
        self.assertEqual(prepared_validation[0]["completion"], [self.messages[-1]])

    def test_chat_template_representation_has_one_generation_prefix_and_eos(self) -> None:
        def template(messages, *, tokenize, add_generation_prompt):
            text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
            if add_generation_prompt:
                return text + "<assistant>"
            return text + ("<eos>" if messages[-1]["role"] == "assistant" else "")

        value = train.prompt_completion(self.messages)
        prompt = template(value["prompt"], tokenize=False, add_generation_prompt=True)
        full = template(value["prompt"] + value["completion"], tokenize=False, add_generation_prompt=False)
        self.assertNotIn(self.messages[-1]["content"], prompt)
        self.assertIn(self.messages[-1]["content"], full)
        self.assertTrue(prompt.endswith("<assistant>"))
        self.assertTrue(full.endswith("<eos>"))
        self.assertEqual(full, train.format_messages(self.messages, template))

    def test_invalid_last_message_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "end with an assistant"):
            train.prompt_completion(self.messages[:-1])

    def test_run_uses_completion_only_loss(self) -> None:
        source = inspect.getsource(train.run)
        self.assertIn("completion_only_loss=True", source)
        self.assertIn("prepare_dataset(load_dataset", source)
        self.assertIn("eval_dataset=validation_dataset", source)
        self.assertIn('eval_strategy="epoch" if validation_dataset is not None else "no"', source)
        self.assertIn('metric_for_best_model="eval_loss"', source)
        self.assertIn('save_strategy="epoch" if validation_dataset is not None else "no"', source)
        self.assertIn("per_device_eval_batch_size=config.eval_batch_size", source)
        self.assertIn("EarlyStoppingCallback", source)
        self.assertNotIn("test.jsonl", source)

    def test_standalone_validation_argument_is_optional(self) -> None:
        arguments = train.build_parser().parse_args(
            ["--dataset", "train.jsonl", "--output", "adapter", "--config", "config.json"]
        )
        self.assertIsNone(arguments.validation)

    def test_validation_metrics_use_the_existing_jsonl_event(self) -> None:
        with patch.object(train, "emit") as emit:
            train.emit_metrics({"eval_loss": 0.25, "eval_runtime": 1.5}, step=12, epoch=2.0)

        emit.assert_called_once_with(
            "metrics",
            step=12,
            loss=None,
            lr=None,
            epoch=2.0,
            eval_loss=0.25,
            eval_runtime=1.5,
        )

    def test_best_model_summary_uses_trainer_state(self) -> None:
        trainer = type(
            "Trainer",
            (),
            {
                "state": type(
                    "State",
                    (),
                    {
                        "best_metric": 0.25,
                        "best_model_checkpoint": "adapter/checkpoint-8",
                        "log_history": [
                            {"eval_loss": 0.25, "epoch": 1.0, "step": 4},
                            {"eval_loss": 0.25, "epoch": 2.0, "step": 8},
                        ],
                    },
                )()
            },
        )()
        self.assertEqual(
            train.best_model_summary(trainer),
            {"best_epoch": 2, "best_eval_loss": 0.25, "best_checkpoint": "adapter/checkpoint-8"},
        )

    def test_training_config_has_safe_eval_and_optional_early_stopping(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{}", encoding="utf-8")
            config = train.read_config(path)
            self.assertEqual(config.eval_batch_size, 1)
            self.assertIsNone(config.early_stopping_patience)
            self.assertEqual(config.early_stopping_threshold, 0)

            path.write_text(
                json.dumps({"eval_batch_size": 2, "early_stopping_patience": 3, "early_stopping_threshold": 0.01}),
                encoding="utf-8",
            )
            config = train.read_config(path)
            self.assertEqual(config.eval_batch_size, 2)
            self.assertEqual(config.early_stopping_patience, 3)
            self.assertEqual(config.early_stopping_threshold, 0.01)
