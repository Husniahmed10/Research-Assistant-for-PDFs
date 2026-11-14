import type { Citation, Message } from "./types";
export function findCitation(citations: Citation[], label: string) {
  return citations.find((citation) => citation.source_ids.includes(label));
}
export function hasSavedAnswer(messages: Message[], previousIds: Set<string>, query: string) {
  const added = messages.filter((message) => !previousIds.has(message.id));
  return added.some(
    (message, index) =>
      message.role === "user" &&
      message.content === query &&
      added[index + 1]?.role === "assistant",
  );
}
export const terminalEvaluation = (status?: string) => !!status && status !== "pending";
export function safeHref(url: string) {
  return /^(https?:\/\/|mailto:|#)/i.test(url) ? url : "";
}
