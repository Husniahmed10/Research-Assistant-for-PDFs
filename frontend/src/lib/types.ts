export type PaperStatus = "uploading" | "processing" | "ready" | "failed";
export interface Paper {
  id: string;
  filename: string;
  file_size_kb: number;
  page_count: number | null;
  status: PaperStatus;
  created_at: string;
}
export const summarySections = [
  ["title_and_authors", "Title & authors"],
  ["abstract", "Abstract"],
  ["problem_statement", "Research problem"],
  ["methodology", "Methodology"],
  ["key_results", "Key results"],
  ["conclusion", "Conclusion"],
] as const;
export type Summary = Record<(typeof summarySections)[number][0], string>;
export interface Citation {
  id: string;
  chunk_id: string;
  page_num: number;
  text: string;
  page: number;
  snippet: string;
  source_ids: string[];
  chunk_ids: string[];
}
export interface Message {
  id: string;
  role: string;
  content: string;
  citations: Citation[];
  model_used: string | null;
  latency_ms: number | null;
  created_at: string;
}
export interface Session {
  session_id: string;
  paper_id: string;
  created_at: string;
}
export interface LocalSession extends Session {
  title: string;
}
export type EvaluationStatus = "pending" | "completed" | "failed" | "skipped" | "not_scheduled";
export interface Evaluation {
  message_id: string;
  status: EvaluationStatus;
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
  overall_score: number | null;
  label: string | null;
  precision_metric: string | null;
  evaluated_at: string | null;
  error_code?: string | null;
  error_message?: string | null;
}
export interface QualityReport {
  total_messages: number;
  total_answers: number;
  evaluated: number;
  pending: number;
  failed: number;
  skipped: number;
  not_scheduled: number;
  avg_faithfulness: number | null;
  avg_relevancy: number | null;
  avg_precision: number | null;
  avg_overall: number | null;
  label: string | null;
  precision_metrics: Record<string, number>;
  score_history: { message_id: string; overall_score: number | null; created_at: string }[];
  avg_latency_ms?: number | null;
  total_questions?: number;
  model_usage?: Record<string, number>;
}
export type StreamEvent =
  | { type: "step" | "token"; content: string }
  | {
      type: "done";
      citations: Citation[];
      model_used: string;
      message_id: string | null;
      latency_ms: number;
    }
  | { type: "error"; content: string; status_code: number };
