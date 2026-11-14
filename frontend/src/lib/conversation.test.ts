import { describe, expect, it } from "vitest";
import { findCitation, hasSavedAnswer, safeHref, terminalEvaluation } from "./conversation";
import { errorDetail, validateUpload } from "./api";
import type { Citation, Message } from "./types";
const citation: Citation = {
  id: "1",
  chunk_id: "c",
  page: 8,
  page_num: 8,
  text: "Evidence",
  snippet: "Evidence",
  source_ids: ["1", "3"],
  chunk_ids: ["c", "d"],
};
const message = (id: string, role: string, content: string): Message => ({
  id,
  role,
  content,
  citations: [],
  created_at: "2026-11-13",
  model_used: null,
  latency_ms: null,
});
describe("conversation reconciliation", () => {
  it("recognizes a newly saved pair after stream interruption", () => {
    expect(
      hasSavedAnswer(
        [
          message("old", "user", "q"),
          message("u", "user", "q"),
          message("a", "assistant", "answer"),
        ],
        new Set(["old"]),
        "q",
      ),
    ).toBe(true);
  });
  it("does not mistake an older identical question for a new saved answer", () => {
    expect(
      hasSavedAnswer(
        [message("old-u", "user", "q"), message("old-a", "assistant", "answer")],
        new Set(["old-u", "old-a"]),
        "q",
      ),
    ).toBe(false);
  });
  it("requires a matching user question and following assistant", () => {
    expect(hasSavedAnswer([message("u", "user", "q")], new Set(), "q")).toBe(false);
    expect(
      hasSavedAnswer(
        [message("u", "user", "other"), message("a", "assistant", "answer")],
        new Set(),
        "q",
      ),
    ).toBe(false);
  });
});
describe("evidence and evaluation", () => {
  it("matches grouped source labels without using array position", () => {
    expect(findCitation([citation], "3")?.page_num).toBe(8);
    expect(findCitation([citation], "2")).toBeUndefined();
  });
  it("stops polling at every terminal evaluation state", () => {
    for (const state of ["completed", "failed", "skipped", "not_scheduled"])
      expect(terminalEvaluation(state)).toBe(true);
    expect(terminalEvaluation("pending")).toBe(false);
    expect(terminalEvaluation()).toBe(false);
  });
  it("rejects unsafe and relative URL protocols", () => {
    for (const href of [
      "javascript:alert(1)",
      "data:text/html,x",
      "//evil.test",
      "/uploads/file.pdf",
    ])
      expect(safeHref(href)).toBe("");
    expect(safeHref("https://example.com")).toBe("https://example.com");
    expect(safeHref("#source-3")).toBe("#source-3");
  });
});
describe("upload and validation feedback", () => {
  it("rejects invalid types, empty files, and oversized PDFs", () => {
    expect(validateUpload({ name: "a.txt", type: "text/plain", size: 100 })).toMatch(/PDF/);
    expect(validateUpload({ name: "a.pdf", type: "application/pdf", size: 0 })).toMatch(/empty/);
    expect(
      validateUpload({ name: "a.pdf", type: "application/pdf", size: 51 * 1024 * 1024 }),
    ).toMatch(/smaller/);
    expect(validateUpload({ name: "A.PDF", type: "application/pdf", size: 1024 })).toBeNull();
  });
  it("formats FastAPI strings and structured validation errors", () => {
    expect(errorDetail({ detail: "Paper is still processing" })).toBe("Paper is still processing");
    expect(errorDetail({ detail: [{ loc: ["body", "paper_id"], msg: "Invalid UUID" }] })).toBe(
      "paper_id: Invalid UUID",
    );
  });
});
