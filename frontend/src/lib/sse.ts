import { API_BASE, ApiError, errorDetail } from "./api";
import type { StreamEvent } from "./types";

/** SSE framing is independent of transport chunk boundaries and UTF-8 decoding. */
export function createSSEParser(onEvent: (value: unknown) => void) {
  let buffer = "";
  let data: string[] = [];
  const line = (value: string) => {
    if (value === "") {
      if (data.length) {
        const payload = data.join("\n");
        data = [];
        onEvent(JSON.parse(payload));
      }
    } else if (value.startsWith("data:")) data.push(value.slice(5).replace(/^ /, ""));
    // Ignore comments and unsupported fields, including keep-alive frames.
  };
  return {
    push(chunk: string) {
      buffer += chunk;
      let index = 0;
      while (index < buffer.length) {
        const char = buffer[index];
        if (char === "\n" || char === "\r") {
          if (char === "\r" && index === buffer.length - 1) break;
          line(buffer.slice(0, index));
          buffer = buffer.slice(index + (char === "\r" && buffer[index + 1] === "\n" ? 2 : 1));
          index = 0;
        } else index++;
      }
    },
    finish() {
      if (buffer.endsWith("\r")) {
        line(buffer.slice(0, -1));
        buffer = "";
      }
      // An unterminated event is not committed. The caller requires a done event.
    },
  };
}
export function parseStreamEvent(value: unknown): StreamEvent {
  if (!value || typeof value !== "object" || !("type" in value))
    throw new Error("Invalid stream event.");
  const event = value as Record<string, unknown>;
  if ((event.type === "step" || event.type === "token") && typeof event.content === "string")
    return event as StreamEvent;
  if (
    event.type === "error" &&
    typeof event.content === "string" &&
    typeof event.status_code === "number"
  )
    return event as StreamEvent;
  if (
    event.type === "done" &&
    Array.isArray(event.citations) &&
    typeof event.model_used === "string" &&
    (typeof event.message_id === "string" || event.message_id === null) &&
    typeof event.latency_ms === "number"
  )
    return event as StreamEvent;
  throw new Error("The API sent an unrecognized stream event.");
}
export async function streamAnswer(
  session: string,
  query: string,
  signal: AbortSignal,
  receive: (event: StreamEvent) => void,
) {
  const response = await fetch(
    `${API_BASE}/api/v1/chat/sessions/${session}/stream?${new URLSearchParams({ query })}`,
    { signal, cache: "no-store", headers: { Accept: "text/event-stream" } },
  );
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    throw new ApiError(
      response.status,
      body ? errorDetail(body) : `Chat request failed (${response.status}).`,
    );
  }
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream"))
    throw new Error("The API did not return an event stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let done = false;
  const parser = createSSEParser((value) => {
    const event = parseStreamEvent(value);
    if (event.type === "error") throw new ApiError(event.status_code, event.content);
    if (event.type === "done") done = true;
    receive(event);
  });
  try {
    while (!done) {
      const chunk = await reader.read();
      if (chunk.done) break;
      parser.push(decoder.decode(chunk.value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.finish();
    if (!done) throw new Error("The connection ended before the answer was confirmed.");
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
