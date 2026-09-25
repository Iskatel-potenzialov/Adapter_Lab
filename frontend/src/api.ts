export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export type Health = {
  status: "ok";
  gpu_state: "idle" | "inference" | "training" | "evaluation";
  model_path_configured: boolean;
  model_path_exists: boolean;
  model_loaded: boolean;
};

export type FieldProfile = {
  observed_types?: string[];
  null_count?: number;
  null_rate?: number;
};

export type DatasetProfile = {
  task_profile?: string;
  evaluation_profile?: string;
  evaluator?: string;
  n_examples?: number;
  n_groups?: number;
  target_schema?: {
    fields: string[];
    field_stats: Record<string, FieldProfile>;
  };
  null_statistics?: {
    total_null_values: number;
    examples_with_null: number;
    examples_without_null: number;
  };
};

export type ValidationResult = {
  valid: boolean;
  errors?: Array<{ line?: number; message?: string; error?: string }>;
  profile?: DatasetProfile;
  n_examples?: number;
  n_groups?: number;
  splits?: Record<"train" | "validation" | "test", ValidationResult>;
};

export class ApiRequestError extends Error {
  constructor(message: string, public readonly validation?: ValidationResult) {
    super(message);
  }
}

export type Dataset = {
  id: string;
  dataset_id?: string;
  dataset_mode?: "canonical" | "predefined";
  size_bytes?: number;
  validation?: ValidationResult;
  split_mode?: string;
  source_count?: number;
  source_group_count?: number;
  train_count?: number;
  validation_count?: number;
  test_count?: number;
  train_group_count?: number;
  validation_group_count?: number;
  test_group_count?: number;
  seed?: number;
  source_hash?: string;
  train_hash?: string;
  validation_hash?: string;
  test_hash?: string;
  dataset_lineage?: Record<string, unknown>;
  profile?: DatasetProfile;
  split?: {
    seed?: number;
    train_count?: number;
    validation_count?: number;
    test_count?: number;
    train_group_count?: number;
    validation_group_count?: number;
    test_group_count?: number;
  };
};

export type DatasetUploadResponse = { id: string; validation: ValidationResult };
export type PredefinedUploadResponse = {
  id: string;
  validation: ValidationResult;
  splits: Record<"train" | "validation" | "test", ValidationResult>;
  dataset: Dataset;
};
export type DatasetSplitResponse = Dataset;

export type TrainingConfig = {
  epochs: number;
  batch_size: number;
  eval_batch_size: number;
  grad_accum: number;
  learning_rate: number;
  lr_scheduler_type: string;
  warmup_ratio: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  max_seq_length: number;
  early_stopping_patience: number | null;
  early_stopping_threshold: number;
};

export type TrainingJob = {
  job_id: string;
  status: string;
  dataset: string;
  created_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  current_step?: number | null;
  progress?: number | null;
  epoch?: number | null;
  loss?: number | null;
  learning_rate?: number | null;
  eval_loss?: number | null;
  eval_runtime?: number | null;
  best_epoch?: number | null;
  best_eval_loss?: number | null;
  best_checkpoint?: string | null;
  adapter_path?: string | null;
  metadata_error?: string | null;
  error?: string | null;
  events?: Array<Record<string, unknown>>;
};

export type Adapter = {
  adapter_id: string;
  valid: boolean;
  validation_error?: string | null;
  metadata_available: boolean;
  metadata_error?: string | null;
  created_at?: string;
  size_bytes?: number;
  files?: string[];
  training_job_id?: string | null;
  dataset_id?: string | null;
  dataset_lineage?: Record<string, unknown> | null;
  training_config?: Record<string, unknown> | null;
  best_epoch?: number | null;
  best_eval_loss?: number | null;
  best_checkpoint?: string | null;
};

export type InferenceResponse = {
  text: string;
  adapter_id: string | null;
  json_valid: boolean;
  parsed_json: unknown | null;
};

export type EvaluationJob = {
  evaluation_id: string;
  dataset_id: string;
  adapter_id: string;
  status: string;
  phase?: string;
  processed?: number;
  total?: number;
  test_count?: number;
  test_hash?: string;
  error?: string | null;
  aggregate?: EvaluationAggregate | null;
};
export type EvaluationResult = { index: number; input_messages: Array<{ role: string; content: string }>; expected: string; base_answer: string; lora_answer: string; base_correct: boolean; lora_correct: boolean; evaluator?: string; judge_valid?: boolean; judge_error?: string | null; judge_order?: Record<string, string>; judge_raw?: string; judge_reason?: string | null; judge_preferred?: string; calculated_preferred?: string; preference_agreement?: boolean; base_judge?: Record<string, number | boolean | null> | null; lora_judge?: Record<string, number | boolean | null> | null };

export type EvaluationAggregate = {
  total_examples: number;
  base_correct: number;
  lora_correct: number;
  both_correct: number;
  both_wrong: number;
  lora_improved: number;
  lora_regressed: number;
  base_accuracy: number;
  lora_accuracy: number;
  absolute_uplift: number;
  structured_json?: {
    base: StructuredMetrics;
    lora: StructuredMetrics;
    per_field: Record<string, PerFieldMetrics>;
  };
  judge?: JudgeMetrics;
  expert?: JudgeMetrics;
};

export type JudgeMetrics = {
  rubric?: string;
  evaluation_profile?: string;
  judge: { provider: string; type: string; passes: number; temperature: number };
  total_examples: number;
  valid_judgements: number;
  invalid_judgements: number;
  pass_threshold: number;
  base: { mean_score: number | null; pass_count: number; pass_rate: number | null };
  lora: { mean_score: number | null; pass_count: number; pass_rate: number | null };
  mean_uplift: number | null;
  paired?: {
    total_examples: number;
    base_correct: number;
    lora_correct: number;
    both_correct: number;
    both_wrong: number;
    lora_improved: number;
    lora_regressed: number;
    base_accuracy: number;
    lora_accuracy: number;
    absolute_uplift: number;
  };
  judge_preference: Record<string, number>;
  calculated_preference: Record<string, number>;
  preference_agreement_rate: number | null;
  per_criterion: Record<string, { examples: number; base_mean_score: number | null; lora_mean_score: number | null }>;
};

export type StructuredMetrics = {
  json_valid_rate?: number;
  full_record_accuracy?: number;
  field_accuracy?: number;
};
export type PerFieldMetrics = {
  base_accuracy: number;
  lora_accuracy: number;
  base_correct: number;
  lora_correct: number;
  total: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new Error(`Backend is unavailable at ${API_BASE_URL}`);
  }
  const body = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.error?.message || body?.detail || `Request failed (${response.status})`;
    const errors = body?.validation?.errors;
    const suffix = Array.isArray(errors) && errors.length ? `: ${formatValidationErrors(errors)}` : "";
    throw new ApiRequestError(`${detail}${suffix}`, body?.validation);
  }
  return body as T;
}

export function formatValidationErrors(errors: Array<{ line?: number; message?: string; error?: string }>): string {
  return errors.map((item) => `line ${item.line ?? "?"}: ${item.message || item.error || "invalid record"}`).join("; ");
}

export const getHealth = () => request<Health>("/health");
type ApiDataset = Omit<Dataset, "id"> & { id?: string; dataset_id?: string };

function normalizeDataset(dataset: ApiDataset): Dataset {
  const split = dataset.split || {};
  return { ...dataset, ...split, id: dataset.id || dataset.dataset_id || "" };
}

export const listDatasets = async () => (await request<ApiDataset[]>("/api/datasets")).map(normalizeDataset);
export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const data = new FormData();
  data.append("file", file);
  return request<DatasetUploadResponse>("/api/datasets/upload", { method: "POST", body: data });
}
export async function uploadPredefinedDataset(train: File, validation: File, test: File): Promise<PredefinedUploadResponse> {
  const data = new FormData();
  data.append("train", train);
  data.append("validation", validation);
  data.append("test", test);
  const response = await request<PredefinedUploadResponse>("/api/datasets/upload-predefined", { method: "POST", body: data });
  return { ...response, dataset: normalizeDataset(response.dataset) };
}
export const splitDataset = async (id: string, payload: { train_ratio: number; validation_ratio: number; test_ratio: number; seed: number }) =>
  normalizeDataset(await request<ApiDataset>(`/api/datasets/${encodeURIComponent(id)}/split`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }));
export const startTraining = (dataset_id: string, config: TrainingConfig) =>
  request<TrainingJob>("/training/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset_id, ...config }) });
export const getTraining = (id: string) => request<TrainingJob>(`/training/${encodeURIComponent(id)}`);
export const listAdapters = () => request<Adapter[]>("/adapters");
export const getAdapter = (id: string) => request<Adapter>(`/adapters/${encodeURIComponent(id)}`);
export const runInference = (payload: { prompt?: string; messages?: Array<{ role: "system" | "user"; content: string }>; adapter_id: string | null; max_new_tokens: number; temperature: number }) =>
  request<InferenceResponse>("/inference/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
export const startEvaluation = (dataset_id: string, adapter_id: string, max_new_tokens: number) =>
  request<EvaluationJob>("/evaluations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset_id, adapter_id, max_new_tokens }) });
export const getEvaluation = (id: string) => request<EvaluationJob>(`/evaluations/${encodeURIComponent(id)}`);
export const getEvaluationResults = (id: string) => request<EvaluationResult[]>(`/evaluations/${encodeURIComponent(id)}/results`);
export const adapterDownloadUrl = (id: string) => `${API_BASE_URL}/adapters/${encodeURIComponent(id)}/download`;
