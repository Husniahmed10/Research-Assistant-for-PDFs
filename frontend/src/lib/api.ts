import type { Evaluation, Paper, Session, Summary, Message, QualityReport } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  "",
);
export const MAX_FILE_SIZE_MB =
  Number(process.env.NEXT_PUBLIC_MAX_FILE_SIZE_MB) > 0
    ? Number(process.env.NEXT_PUBLIC_MAX_FILE_SIZE_MB)
    : 50;
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
export function errorDetail(body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail))
      return detail
        .map(
          (item: { loc?: unknown[]; msg?: string }) =>
            `${item.loc?.filter((part) => part !== "body").join(" · ") || "Request"}: ${item.msg || "Invalid value"}`,
        )
        .join("; ");
  }
  return "The request could not be completed. Please try again.";
}
export const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : "Something went wrong. Please try again.";

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, cache: "no-store" });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new ApiError(
      0,
      "Cannot reach the research API. Check that the backend is running and try again.",
    );
  }
  if (!response.ok) {
    let detail: unknown;
    try {
      detail = await response.json();
    } catch {
      /* Non-JSON proxy error. */
    }
    throw new ApiError(
      response.status,
      detail ? errorDetail(detail) : `The API returned ${response.status}. Please try again.`,
    );
  }
  return response.json() as Promise<T>;
}
export const api = {
  papers: (signal?: AbortSignal) => request<Paper[]>("/api/v1/papers/", { signal }),
  status: (id: string, signal?: AbortSignal) =>
    request<{ paper_id: string; status: Paper["status"]; page_count: number | null }>(
      `/api/v1/papers/${id}/status`,
      { signal },
    ),
  summary: (id: string, signal?: AbortSignal) =>
    request<Summary>(`/api/v1/papers/${id}/summary`, { signal }),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ paper_id: string; filename: string; status: string }>(
      "/api/v1/papers/upload",
      { method: "POST", body },
    );
  },
  delete: (id: string) =>
    request<{ message: string }>(`/api/v1/papers/${id}`, { method: "DELETE" }),
  createSession: (paper_id: string) =>
    request<Session>("/api/v1/chat/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_id }),
    }),
  messages: (id: string, signal?: AbortSignal) =>
    request<Message[]>(`/api/v1/chat/sessions/${id}/messages`, { signal }),
  evaluation: async (id: string, signal?: AbortSignal): Promise<Evaluation> => {
    try {
      return await request<Evaluation>(`/api/v1/evaluation/messages/${id}`, { signal });
    } catch (error) {
      if (error instanceof ApiError && error.status === 404)
        return {
          message_id: id,
          status: "not_scheduled",
          faithfulness: null,
          answer_relevancy: null,
          context_precision: null,
          context_recall: null,
          overall_score: null,
          label: null,
          precision_metric: null,
          evaluated_at: null,
        };
      throw error;
    }
  },
  report: (kind: "papers" | "sessions", id: string, signal?: AbortSignal) =>
    request<QualityReport>(
      `/api/v1/evaluation/${kind}/${id}/${kind === "papers" ? "report" : "summary"}`,
      { signal },
    ),
};
export function validateUpload(file: Pick<File, "name" | "type" | "size">): string | null {
  if (!file.name.toLowerCase().endsWith(".pdf") || (file.type && file.type !== "application/pdf"))
    return "Choose a PDF file (.pdf).";
  if (!file.size) return "This file is empty. Choose another PDF.";
  if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024)
    return `The PDF must be ${MAX_FILE_SIZE_MB} MB or smaller.`;
  return null;
}
