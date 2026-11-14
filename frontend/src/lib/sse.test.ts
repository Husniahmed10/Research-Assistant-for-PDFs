import { afterEach, describe, expect, it, vi } from "vitest";
import { createSSEParser, parseStreamEvent, streamAnswer } from "./sse";
import type { StreamEvent } from "./types";
describe("SSE framing", () => {
  it("handles split CRLF boundaries, heartbeats, and multiple events", () => {
    const received: unknown[] = [];
    const parser = createSSEParser((value) => received.push(value));
    const wire =
      ': keep-alive\r\n\r\ndata: {"type":"step","content":"Search"}\r\n\r\ndata: {"type":"token","content":" a"}\n\n';
    for (const char of wire) parser.push(char);
    parser.finish();
    expect(received).toEqual([
      { type: "step", content: "Search" },
      { type: "token", content: " a" },
    ]);
  });
  it("joins multiple data lines and ignores SSE metadata", () => {
    const receive = vi.fn();
    const parser = createSSEParser(receive);
    parser.push('event: message\nid: 9\ndata: {"type":"token",\ndata: "content":"Hello"}\n\n');
    expect(receive).toHaveBeenCalledWith({ type: "token", content: "Hello" });
  });
  it("does not commit an unterminated frame", () => {
    const receive = vi.fn();
    const parser = createSSEParser(receive);
    parser.push('data: {"type":"token","content":"partial"}');
    parser.finish();
    expect(receive).not.toHaveBeenCalled();
  });
  it("rejects malformed JSON and unknown event shapes", () => {
    expect(() => createSSEParser(() => {}).push("data: nope\n\n")).toThrow();
    expect(() => parseStreamEvent({ type: "token", content: 5 })).toThrow();
    expect(() => parseStreamEvent({ type: "done", citations: [] })).toThrow();
  });
});
function response(text: string) {
  const bytes = new TextEncoder().encode(text);
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );
}
afterEach(() => vi.unstubAllGlobals());
describe("stream transport", () => {
  it("decodes split UTF-8 and preserves token spacing", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          response(
            'data: {"type":"token","content":" Résumé 🔍"}\n\ndata: {"type":"done","citations":[],"model_used":"test","message_id":"1","latency_ms":10}\n\n',
          ),
        ),
    );
    const events: StreamEvent[] = [];
    await streamAnswer("session", "question & more?", new AbortController().signal, (value) =>
      events.push(value),
    );
    expect(events[0]).toEqual({ type: "token", content: " Résumé 🔍" });
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("query=question+%26+more%3F");
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("treats EOF before done as an interruption without reconnecting", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(response('data: {"type":"token","content":"Partial"}\n\n')),
    );
    await expect(streamAnswer("s", "q", new AbortController().signal, () => {})).rejects.toThrow(
      "before the answer was confirmed",
    );
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("surfaces in-band server errors without replaying", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          response('data: {"type":"error","content":"Verification failed","status_code":502}\n\n'),
        ),
    );
    await expect(streamAnswer("s", "q", new AbortController().signal, () => {})).rejects.toThrow(
      "Verification failed",
    );
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("surfaces HTTP validation errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: [{ loc: ["body", "query"], msg: "Too long" }] }), {
          status: 422,
        }),
      ),
    );
    await expect(streamAnswer("s", "q", new AbortController().signal, () => {})).rejects.toThrow(
      "query: Too long",
    );
  });
});
