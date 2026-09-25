# Qwen LoRA Fine-tuning Studio

Локальная веб-студия для подготовки JSONL-датасетов, text-only QLoRA-обучения
`Qwen2.5-VL-7B-Instruct` и проверки, что LoRA-адаптер действительно улучшает
результат относительно clean Base model.

> В версии v1 используется только текстовая часть модели.  

V1 ориентирован на structured transformation: модель извлекает и возвращает
факты из входных данных в стабильном JSON-формате. 

---

## Что делает Studio

Studio проводит один воспроизводимый pipeline:

```text
Dataset → Validation → Split → Training → LoRA Adapter → Inference → Base vs LoRA Evaluation
```

| Действие пользователя | Что делает Studio |
|---|---|
| Загрузить один полный JSONL | Валидирует dataset и создаёт deterministic canonical Train / Validation / Test split. |
| Загрузить готовые Train / Validation / Test | Валидирует три файла, проверяет group leakage и сохраняет их без перемешивания. |
| Запустить training | Создаёт job и запускает отдельный `backend/train.py` subprocess с QLoRA. |
| Открыть Base или LoRA в Chat | Загружает одну конфигурацию в GPU и возвращает text-only generation. |
| Запустить Base vs LoRA Evaluation | Сравнивает ответы на одном Test, сохраняет per-example results и aggregate metrics. |
| Скачать adapter | Формирует ZIP только с deployable LoRA-файлами, без training checkpoints. |

Studio отвечает не только на вопрос «модель генерирует JSON?», но и на вопрос
«стала ли LoRA точнее извлекать нужные поля на независимом Test?».

---

## Архитектура

```text
Windows
React + TypeScript + Vite
        │
        │ REST + WebSocket
        ▼
Ubuntu
FastAPI
        │
        ├── text inference
        │
        └── train.py subprocess
                │
                ▼
        Qwen2.5-VL-7B-Instruct
        + QLoRA adapters
```


### Как проходит один запуск

1. Пользователь выбирает Automatic или Predefined dataset mode.
2. Backend валидирует JSONL, фиксирует profile, hashes и split metadata.
3. Training API запускает `train.py`; FastAPI не обучает модель внутри HTTP request.
4. `train.py` пишет structured JSON Lines events, checkpoints и LoRA adapter.
5. Inference и evaluation последовательно загружают clean Base и Base + LoRA.
6. Evaluation сохраняет ответы, deterministic metrics и, для structured JSON,
   blind LLM Judge result.

### Почему training — отдельный subprocess

Обучение длительное, использует GPU и может завершиться с CUDA error. FastAPI
создаёт job, формирует config и наблюдает процесс; `train.py` выполняет обучение
и завершается с ненулевым exit code при ошибке. Завершение без события `done` не
считается успешным job.

---

## Ограничения v1

- одна GPU;
- целевая VRAM: 12 ГБ;
- только текст;
- одно обучение одновременно;
- training и inference не выполняются одновременно;
- модель и adapter загружаются локально;
- Hugging Face network access не требуется;
- данные хранятся в файлах, без БД.

### Почему training и inference взаимно исключены

Целевая RTX 3060 имеет 12 GB VRAM. Одновременная загрузка Qwen для training и
inference небезопасна. В одном backend process действует простое in-memory GPU
reservation: training, generation и evaluation не используют GPU параллельно;
перед training/evaluation idle inference-модель выгружается.

---

## Требования

### Ubuntu

- Python 3.10;
- NVIDIA GPU;
- CUDA-compatible PyTorch;
- минимум 32 ГБ RAM рекомендуется;
- локально скачанная `Qwen2.5-VL-7B-Instruct`.

---

## Backend

Для проверенного Ubuntu GPU-окружения:

```bash
conda activate qwen_studio
cd backend
python -m pip check
uvicorn server:app --host 0.0.0.0 --port 8000
```


### Verified Stage 3 GPU environment

Smoke test QLoRA успешно проверен на Ubuntu в Conda environment `qwen_studio`:

- Python 3.10.21; NVIDIA GeForce RTX 3060 12 GB; driver 535.309.01;
  `nvidia-smi` CUDA 12.2;
- PyTorch 2.5.1+cu121 (CUDA runtime 12.1);
- Transformers 4.49.0; bitsandbytes 0.45.0; Accelerate 1.3.0;
- PEFT 0.21.0; TRL 0.17.0; Datasets 5.0.1;
- FastAPI 0.139.0; python-multipart 0.0.32.

В этом окружении модель загрузилась локально в 4-bit NF4, text-only QLoRA
прошёл 8 steps без CUDA OOM и сохранил LoRA adapter. Не обновляйте эти версии
без отдельной проверки на GPU.

---

## Frontend


Frontend по умолчанию работает через Vite. Скопируйте `frontend/.env.example` в
`frontend/.env` и при необходимости укажите адрес Ubuntu backend, доступный из
браузера:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Для стандартного Vite origin `http://localhost:5173` backend использует значение
`FRONTEND_ORIGIN` из `backend/.env`. Если frontend запускается с другого origin,
укажите этот exact origin в `FRONTEND_ORIGIN` и перезапустите backend.

Dashboard поддерживает полный v1 workflow. Для inference отправляется либо
legacy `prompt`, либо system/user messages — не оба варианта одновременно.

---

## Проверка backend

После запуска backend:

```text
GET /health
```

Ожидается JSON со статусом backend и GPU/runtime state. Поля
`model_path_configured`, `model_path_exists` и `model_loaded` позволяют отличить
работающий backend от готовности локальной модели без её загрузки.

---

## Dataset: Automatic и Predefined modes

### Automatic split

Пользователь загружает один полный JSONL через `POST /api/datasets/upload`, затем
создаёт canonical Train / Validation / Test через `POST /api/datasets/{id}/split`.
По умолчанию используются ratios `0.8 / 0.1 / 0.1` и `seed=42`.

Если все examples содержат `group_id`, split распределяет целые группы: варианты
одной исходной сущности не попадут одновременно в Train и Test. Если `group_id`
нет ни у одного example, распределяются отдельные строки. Смешанный формат
отклоняется, чтобы не допустить неявный leakage.

### Predefined Train / Validation / Test

Пользователь загружает готовые `train.jsonl`, `validation.jsonl` и `test.jsonl`
через `POST /api/datasets/upload-predefined`.

- каждый файл проходит обычную JSONL validation;
- `group_id` обязателен в каждой строке;
- одна группа не может пересекать Train, Validation и Test;
- порядок строк сохраняется, пользовательские splits не перемешиваются;
- для каждого split сохраняются row count, group count и hash.

Оба режима приводят к одной рабочей форме: Training читает только Train и
Validation, а Base vs LoRA Evaluation — только Test.

### Формат JSONL

Формат:

```text
.jsonl
```

Одна строка = один training example. Необязательные top-level `group_id` и
`evaluator` — metadata: `group_id` используется для canonical split и
сопоставления evaluation, а `evaluator` задаёт способ deterministic evaluation;
оба не передаются модели.

Пример:

```jsonl
{"messages": [{"role": "system", "content": "Ты — справочник ОКВЭД-2. Отвечай точно, используя только официальные коды и наименования."}, {"role": "user", "content": "ОКВЭД 81.2 — что это?"}, {"role": "assistant", "content": "{\"code\": \"81.2\", \"name\": \"Деятельность по чистке и уборке\", \"section\": \"N\"}"}]}
{"group_id": "opaque-group-key", "messages": [{"role": "user", "content": "Вопрос"}, {"role": "assistant", "content": "Ответ"}]}
{"evaluator": "normalized_text", "messages": [{"role": "user", "content": "Вопрос"}, {"role": "assistant", "content": "Ответ"}]}
{"messages": [{"role": "system", "content": "Ты — справочник ОКВЭД-2. Отвечай точно, используя только официальные коды и наименования."}, {"role": "user", "content": "Дай название вида деятельности по коду ОКВЭД: 50.10.31"}, {"role": "assistant", "content": "Аренда морских судов заграничного плавания для перевозки пассажиров с экипажем"}]}
```

Правила:

- `messages` — список;
- `content` всегда строка;
- допустимые роли:
  - `system`
  - `user`
  - `assistant`
- последнее сообщение — `assistant`.
- если `group_id` присутствует, это непустая строка.
- если `evaluator` присутствует, это одна из строк: `exact`, `normalized_text`, `json`.

Если ожидаемый ответ является JSON, JSON хранится как строка внутри `assistant.content`.

### Structured JSON profile

Для v1 structured JSON dataset каждая строка дополнительно объявляет
`"task_profile":"structured_json"`, использует `"evaluator":"json"` и имеет
непустой `group_id`. `assistant.content` должен содержать JSON object с одним и
тем же набором top-level полей во всех examples. JSON `null` допустим и отличен
от отсутствующего поля. При upload сохраняются schema и null statistics; split
сохраняет их в dataset lineage.

Для этого профиля deterministic evaluation отвечает на разные вопросы:

| Метрика | Что показывает |
|---|---|
| JSON validity | Получился ли JSON object, пригодный для сравнения со схемой. |
| Exact full-record accuracy | Совпадает ли весь object с expected target. |
| Field accuracy | Какая доля полей совпала с expected. |
| Per-field accuracy | В каких конкретных полях Base и LoRA точны. |

Raw JSON validity хранится отдельно от normalized validity. Для вывода модели
безопасно распознаются только однозначные внешние обёртки валидного JSON
object/array: `JSON:`, fenced markdown JSON block, короткий prose prefix или
один внешний JSON string. Studio не исправляет quoted keys, single quotes,
trailing commas, незакрытые braces, broken escapes, missing commas и другие
синтаксические или смысловые ошибки внутри JSON. Исходные ответы Base и LoRA
сохраняются дословно.

```jsonl
{"task_profile":"structured_json","group_id":"record-001","evaluator":"json","messages":[{"role":"user","content":"Name: Ada; Power: 15000 W; Section is absent."},{"role":"assistant","content":"{\"name\":\"Ada\",\"power_kw\":15,\"section\":null}"}]}
```

### LLM Judge

Structured Base-vs-LoRA evaluation сохраняет deterministic JSON/schema/field
metrics и дополнительно выполняет один blind A/B проход LLM Judge для каждого Test
example. Judge получает input, reference target и анонимные `ANSWER A` / `ANSWER B`;
их порядок детерминированно выводится из `group_id` и индекса. Текущий provider —
локальная clean Base Qwen (`local_base`), без внешнего API.

Judge не заменяет deterministic metrics: он дополнительно оценивает semantic
correctness, completeness и format. Invalid/unparseable judgement не считается
ни PASS, ни FAIL; он исключается из quality/pass/preference metrics и учитывается
отдельно как invalid judgement.

Dataset API на backend:

- `POST /api/datasets/upload` — загрузить и провалидировать `.jsonl` (до 500 МБ);
- `POST /api/datasets/upload-predefined` — загрузить готовые `train`, `validation` и
  `test` JSONL. Они сохраняются без перемешивания; для всех строк обязателен
  `group_id`, а группы не могут пересекаться между split;
- `GET /api/datasets` — получить список;
- `GET /api/datasets/{id}/preview` — получить первые примеры;
- `POST /api/datasets/{id}/split` — создать один канонический train/validation/test split.
  По умолчанию используются ratios `0.8/0.1/0.1` и `seed=42`. Повторный запрос с
  той же конфигурацией идемпотентен; другая конфигурация для уже разделённого
  dataset отклоняется. Training использует только сохранённый `train` split;
  validation и test остаются зарезервированными.
  Если все строки содержат `group_id`, целые группы распределяются между split;
  если ни одна — строки распределяются по отдельности. Смешанный формат
  отклоняется при создании split.
- `DELETE /api/datasets/{id}` — удалить dataset.

В UI доступны два режима: автоматическое canonical разделение одного полного
JSONL или загрузка готовых Train / Validation / Test. В обоих режимах Training
использует только Train + Validation, а Base vs LoRA evaluation — только Test.
Для predefined splits `group_id` обязателен в каждой строке; одна группа не может
попасть более чем в один из Train, Validation и Test. Загруженные пользователем
splits сохраняют исходный порядок строк и не перемешиваются.

---

## Каталоги данных

```text
backend/data/datasets/
backend/data/adapters/
backend/data/experiments/
backend/logs/
```

---

## Базовый workflow

1. Запустить backend.
2. Запустить frontend.
3. Открыть страницу Data.
4. Загрузить полный `.jsonl` для automatic split либо готовые Train / Validation /
   Test JSONL для predefined mode.
5. Проверить validation.
6. Для automatic mode создать canonical split через `POST /api/datasets/{id}/split`;
   predefined mode уже содержит рабочие splits.
7. Открыть Training.
8. Выбрать dataset и запустить training (без split запуск отклоняется).
9. Смотреть logs и metrics.
10. Дождаться сохранения LoRA adapter.
11. Открыть Adapters.
12. Активировать adapter.
13. Открыть Chat.
14. Проверить ответы модели.

---

## Training

Training выполняется отдельным:

```text
backend/train.py
```

Backend только управляет subprocess. `train.py` подготавливает prompt/completion
dataset так, что target assistant message не попадает в model input.

Базовые настройки:

```text
4-bit NF4
batch size = 1
gradient accumulation = 16
gradient checkpointing = true
paged_adamw_8bit
max_seq_length = configurable
```

Training использует QLoRA, 4-bit NF4 и PEFT LoRA adapters. Validation использует
отдельный безопасный `eval_batch_size` (default `1`), не влияющий на train batch
size. Training пишет train/eval metrics, сохраняет best checkpoint и финальный
LoRA adapter.

Early stopping выключен по умолчанию; его можно включить через
`early_stopping_patience` (целое число от `1`) и
`early_stopping_threshold` (default `0`). `max_seq_length` — параметр обучения:
его нужно выбирать по длине конкретного датасета, а не считать универсально
равным `256`.

При OOM настройки уменьшаются согласно `SPEC.md`.

### Standalone smoke test на Ubuntu

```bash
conda activate qwen_studio
cd backend
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
python train.py \
  --dataset data/datasets/<dataset_id>/train.jsonl \
  --output data/adapters/<exp_id> \
  --config data/experiments/<exp_id>_config.json
```

Пример config-файла:

```json
{
  "exp_id": "okved_v1",
  "epochs": 3,
  "batch_size": 1,
  "eval_batch_size": 1,
  "grad_accum": 16,
  "learning_rate": 0.0002,
  "max_seq_length": 256,
  "early_stopping_patience": null,
  "early_stopping_threshold": 0
}
```

### Training API

Training API запускает тот же `train.py` отдельным subprocess и хранит текущие jobs
в памяти backend-процесса. Одновременно разрешено одно обучение.

```text
POST /training/start
GET /training
GET /training/{job_id}
```

Пример запуска:

```json
{
  "dataset_id": "<dataset_id>",
  "job_id": "okved_v1",
  "epochs": 3,
  "batch_size": 1,
  "grad_accum": 16,
  "learning_rate": 0.0002
}
```

Ответ содержит `job_id`, статус, PID, текущие metrics и `log_path`. После
события `done` статус становится `completed`, а `adapter_path` указывает на
сохранённый LoRA adapter. Если `MODEL_PATH` отсутствует или некорректен, job
завершается с понятной ошибкой, а не выдаётся за completed.

---

## Validation case

Реальный validation experiment: **System analyst requirements structuring**.

| Параметр | Значение |
|---|---:|
| Dataset | 500 examples / 100 groups |
| Train / Validation / Test | 400 / 50 / 50 |
| Base JSON valid | 100% |
| LoRA JSON valid | 100% |
| Base exact full-record accuracy | 0% |
| LoRA exact full-record accuracy | 72% |
| Base field accuracy | 34.33% |
| LoRA field accuracy | 92.00% |

| Field | Base | LoRA |
|---|---:|---:|
| `functional_requirements` | 0% | 80% |
| `non_functional_requirements` | 26% | 90% |
| `ui_requirements` | 50% | 100% |
| `security_requirements` | 70% | 100% |
| `audit_requirements` | 60% | 100% |
| `acceptance_criteria` | 0% | 82% |

Главный вывод: LoRA научилась не просто генерировать JSON — Base уже давала
100% valid JSON. Улучшение связано с более точной классификацией сырого
бизнес-запроса по заданным категориям требований.

### Inference API

Text-only generation доступна на Ubuntu GPU backend:

```text
GET /inference/status
POST /inference/generate
```

`adapter_id: null` выбирает clean Base; безопасный adapter ID выбирает Base +
LoRA. В GPU хранится только одна конфигурация, поэтому переключение выгружает
предыдущую. Inference отклоняется, пока training job queued/running; training
отклоняется, если generation уже использует GPU.

`POST /inference/generate` принимает либо legacy `prompt`, либо text-only
`messages` из `system`/`user` сообщений. В response всегда добавляются
`json_valid` и `parsed_json`; невалидный JSON не делает generation ошибкой.

```json
{
  "prompt": "What is OKVED 81.2?",
  "adapter_id": null,
  "max_new_tokens": 128,
  "temperature": 0
}
```

### Evaluation API

Test split можно сравнить асинхронно для Base и одного LoRA adapter. Base и LoRA
получают одинаковые `input_messages`, canonical Test order и `max_new_tokens`;
обе фазы используют deterministic generation (`temperature: 0`).

Последовательность GPU lifecycle:

```text
Base × N → unload Base → LoRA × N → unload LoRA → Judge × N (structured JSON)
```

Evaluation не публикуется как completed, пока не обработаны все pairs и aggregate
не прошёл integrity checks. Metadata и per-example results сохраняются в
`backend/data/evaluations/`.

```text
POST /evaluations
GET /evaluations
GET /evaluations/{evaluation_id}
GET /evaluations/{evaluation_id}/results
```

```json
{"dataset_id":"<dataset_id>","adapter_id":"<adapter_id>","max_new_tokens":128}
```

### Adapter API

Adapters are local directories under `backend/data/adapters/` and are managed
only by their safe adapter ID:

```text
GET /adapters
GET /adapters/{adapter_id}
DELETE /adapters/{adapter_id}
```

New successful training jobs write a small `metadata.json` beside the adapter
with its dataset ID, training job ID, timestamp and actual training config.
Older adapters without metadata remain valid. A loaded adapter or an adapter
currently being produced by training cannot be deleted.

---

## Offline mode

Модель должна загружаться локально.

Backend/training должны использовать:

```text
HF_HUB_OFFLINE=1
local_files_only=True
```



