import { expect, test, type Page } from "@playwright/test";
const browserErrors = new WeakMap<Page, string[]>();
test.beforeEach(async ({ page }) => {
  const errors: string[] = [];
  browserErrors.set(page, errors);
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (/hydration|a tree hydrated|server.rendered html/i.test(message.text()))
      errors.push(message.text());
  });
});
test.afterEach(async ({ page }) => {
  expect(
    browserErrors.get(page),
    "No application crashes or hydration errors in a clean browser",
  ).toEqual([]);
});
const paperId = "b9f4d908-99c5-4a35-b241-935f1038f2a8";
const sessionId = "6f64d5b8-2b64-45dd-aa35-18aaf38133b9";
const assistantId = "b87137d5-4ac0-481c-8a2f-998ca2c93221";
const citation = {
  id: "1",
  chunk_id: "chunk-1",
  page: 4,
  page_num: 4,
  text: "A controlled experiment compared three retrieval strategies across 120 research questions.",
  snippet: "A controlled experiment",
  source_ids: ["1", "3"],
  chunk_ids: ["chunk-1", "chunk-3"],
};
const answer =
  "The study compares **three retrieval strategies** using a controlled experiment [3].";
const now = "2026-11-13T12:00:00Z";
const report = {
  total_messages: 2,
  total_answers: 1,
  evaluated: 1,
  pending: 0,
  failed: 0,
  skipped: 0,
  not_scheduled: 0,
  avg_faithfulness: 0.92,
  avg_relevancy: 0.88,
  avg_precision: 0.85,
  avg_overall: 0.89,
  label: "Excellent",
  precision_metrics: { context_utilization: 1 },
  score_history: [],
  total_questions: 1,
  avg_latency_ms: 1250,
  model_usage: { "test-model": 1 },
};
async function mockAPI(
  page: Page,
  interrupted = false,
  persistAnswer = true,
  streamGate?: Promise<void>,
) {
  let uploaded = false;
  let checks = 0;
  let asked = false;
  let pending = 0;
  let submissions = 0;
  await page.route("http://localhost:8000/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const respond = (value: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(value),
        headers: { "Access-Control-Allow-Origin": "http://localhost:3000" },
      });
    if (route.request().method() === "OPTIONS")
      return route.fulfill({
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "http://localhost:3000",
          "Access-Control-Allow-Headers": "Content-Type, Accept",
          "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
        },
      });
    if (path === "/") return respond({ status: "ok", app: "PaperLens" });
    if (path === "/api/v1/papers/upload") {
      uploaded = true;
      return respond(
        { paper_id: paperId, filename: "retrieval-study.pdf", status: "processing" },
        202,
      );
    }
    if (path === "/api/v1/papers/")
      return respond(
        uploaded
          ? [
              {
                id: paperId,
                filename: "retrieval-study.pdf",
                file_size_kb: 281,
                page_count: checks++ > 0 ? 9 : null,
                status: checks > 1 ? "ready" : "processing",
                created_at: now,
              },
            ]
          : [],
      );
    if (path.endsWith("/summary") && path.includes("/papers/"))
      return respond({
        title_and_authors: "Retrieval Strategies — A. Researcher",
        abstract: "An investigation into evidence retrieval.",
        problem_statement: "Finding the right evidence is difficult.",
        methodology: "A controlled comparison of three retrieval strategies.",
        key_results: "Hybrid retrieval performed well.",
        conclusion: "Evaluate retrieval with diverse questions.",
      });
    if (path === "/api/v1/chat/sessions")
      return respond({ session_id: sessionId, paper_id: paperId, created_at: now }, 201);
    if (path.endsWith("/messages") && path.includes("/chat/"))
      return respond(
        asked
          ? [
              {
                id: "user-1",
                role: "user",
                content: "Explain the methodology",
                citations: [],
                model_used: null,
                latency_ms: null,
                created_at: now,
              },
              {
                id: assistantId,
                role: "assistant",
                content: answer,
                citations: [citation],
                model_used: "test-model",
                latency_ms: 1250,
                created_at: now,
              },
            ]
          : [],
      );
    if (path.endsWith("/stream")) {
      submissions++;
      if (streamGate) await streamGate;
      asked = persistAnswer;
      const events = [
        { type: "step", content: "Running hybrid search…" },
        { type: "step", content: "Reranking results…" },
        { type: "token", content: answer },
      ];
      return route.fulfill({
        contentType: "text/event-stream",
        body:
          events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") +
          (interrupted
            ? ""
            : `data: ${JSON.stringify({ type: "done", citations: [citation], model_used: "test-model", message_id: assistantId, latency_ms: 1250 })}\n\n`),
        headers: { "Access-Control-Allow-Origin": "http://localhost:3000" },
      });
    }
    if (path.includes("/evaluation/messages/"))
      return respond(
        {
          message_id: assistantId,
          status: pending++ ? "completed" : "pending",
          faithfulness: 0.92,
          answer_relevancy: 0.88,
          context_precision: 0.85,
          context_recall: null,
          overall_score: 0.89,
          label: "Excellent",
          precision_metric: "context_utilization",
          evaluated_at: now,
        },
        pending > 1 ? 200 : 202,
      );
    if (path.startsWith("/api/v1/evaluation/")) return respond(report);
    if (route.request().method() === "DELETE") {
      uploaded = false;
      return respond({ message: "deleted" });
    }
    return respond({ detail: "Not found" }, 404);
  });
  return { submissions: () => submissions };
}
test("switching paper views preserves an in-flight answer and the next question draft", async ({
  page,
}) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const mock = await mockAPI(page, false, true, gate);
  try {
    await uploadAndOpen(page);
    await page.getByRole("button", { name: "Send question" }).click();
    await expect.poll(() => mock.submissions()).toBe(1);
    await page.getByRole("tab", { name: "Answer quality" }).click();
    await expect(page.getByRole("heading", { name: "Research you can examine." })).toBeVisible();
    await page.getByRole("tab", { name: "Summary", exact: true }).click();
    release();
    await page.getByRole("tab", { name: "Chat with paper" }).click();
    await expect(page.getByRole("button", { name: "Source 3, page 4" })).toBeVisible();
    await expect(page.locator(".interrupted")).not.toBeVisible();
    expect(mock.submissions()).toBe(1);
    await page.getByLabel("Ask about this paper").fill("A draft to keep");
    await page.getByRole("tab", { name: "Summary", exact: true }).click();
    await page.getByRole("tab", { name: "Chat with paper" }).click();
    await expect(page.getByLabel("Ask about this paper")).toHaveValue("A draft to keep");
    await page.getByRole("tab", { name: "Summary", exact: true }).click();
    await page.getByRole("button", { name: "What are the limitations?", exact: true }).click();
    await expect(page.getByLabel("Ask about this paper")).toHaveValue("What are the limitations?");
    expect(mock.submissions()).toBe(1);
  } finally {
    release();
  }
});
async function uploadAndOpen(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Upload paper", exact: true }).click();
  await page.getByLabel("Choose PDF", { exact: true }).setInputFiles({
    name: "retrieval-study.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.7\nMock browser-test PDF"),
  });
  await page.getByRole("dialog").getByRole("button", { name: "Upload paper", exact: true }).click();
  await expect(page.locator(".status.processing")).toBeVisible();
  await expect(page.locator(".status.ready")).toBeVisible({ timeout: 15000 });
  await page
    .getByRole("link", { name: /retrieval-study.pdf/ })
    .last()
    .click();
  await expect(page.getByRole("heading", { name: "Methodology", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Explain the methodology", exact: true }).click();
  await page.getByRole("button", { name: "Create conversation" }).click();
  await expect(page.getByLabel("Ask about this paper")).toHaveValue("Explain the methodology");
}
test("upload, process, chat, grouped citation, quality, history, and deletion", async ({
  page,
}, info) => {
  const mock = await mockAPI(page);
  await uploadAndOpen(page);
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByRole("button", { name: "Source 3, page 4" })).toBeVisible();
  await page.getByRole("button", { name: "Source 3, page 4" }).click();
  await expect(page.getByRole("dialog")).toContainText(citation.text);
  await page.getByRole("button", { name: "Close evidence" }).click();
  await expect(page.getByRole("button", { name: /Excellent · 89%/ })).toBeVisible({
    timeout: 12000,
  });
  await page.getByRole("button", { name: /Excellent · 89%/ }).click();
  await expect(
    page.getByText("Context utilization · reference-free", { exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Answer quality" }).click();
  await expect(page.getByText("89%", { exact: true })).toBeVisible();
  await page.screenshot({ path: `test-results/${info.project.name}-quality.png`, fullPage: true });
  await page.reload();
  await page.getByRole("tab", { name: "Chat with paper" }).click();
  await expect(page.getByRole("button", { name: "Source 3, page 4" })).toBeVisible();
  expect(mock.submissions()).toBe(1);
  await expect(page.locator("body")).toHaveJSProperty(
    "scrollWidth",
    await page.locator("body").evaluate((body) => body.clientWidth),
  );
  await page.getByRole("button", { name: "Delete retrieval-study.pdf" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete paper", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Your next insight starts here" })).toBeVisible();
});
test("interrupted stream reconciles a saved answer without replay", async ({ page }) => {
  const mock = await mockAPI(page, true);
  await uploadAndOpen(page);
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByRole("button", { name: "Source 3, page 4" })).toBeVisible();
  expect(mock.submissions()).toBe(1);
  await expect(page.getByText("Connection interrupted", { exact: true })).not.toBeVisible();
});
test("keyboard upload dialog returns focus and theme changes", async ({ page }) => {
  await mockAPI(page);
  await page.goto("/");
  const upload = page.getByRole("button", { name: "Upload paper", exact: true });
  await upload.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(upload).toBeFocused();
  await page.getByRole("button", { name: "Toggle color theme" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
});
test("unsaved interruption blocks submission until history is checked and retry is explicit", async ({
  page,
}) => {
  const mock = await mockAPI(page, true, false);
  await uploadAndOpen(page);
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(
    page.getByRole("alert").getByText("Connection interrupted", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Send question" })).toBeDisabled();
  await page.getByRole("button", { name: "Check saved history", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Edit / explicitly retry", exact: true }),
  ).toBeVisible();
  expect(mock.submissions()).toBe(1);
  await page.getByRole("button", { name: "Edit / explicitly retry", exact: true }).click();
  await expect(page.getByLabel("Ask about this paper")).toHaveValue("Explain the methodology");
  expect(mock.submissions()).toBe(1);
});
