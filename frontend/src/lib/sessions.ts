"use client";
import { useMemo, useSyncExternalStore } from "react";
import { API_BASE } from "./api";
import type { LocalSession } from "./types";
const key = `paperlens:sessions:v1:${API_BASE}`;
const eventName = "paperlens:sessions";
const subscribe = (listener: () => void) => {
  window.addEventListener("storage", listener);
  window.addEventListener(eventName, listener);
  return () => {
    window.removeEventListener("storage", listener);
    window.removeEventListener(eventName, listener);
  };
};
function snapshot() {
  try {
    return localStorage.getItem(key) || "[]";
  } catch {
    return "[]";
  }
}
function parseSessions(raw: string): LocalSession[] {
  try {
    const value: unknown = JSON.parse(raw);
    if (!Array.isArray(value)) return [];
    return value.filter(
      (item): item is LocalSession =>
        item &&
        typeof item.session_id === "string" &&
        typeof item.paper_id === "string" &&
        typeof item.title === "string" &&
        typeof item.created_at === "string",
    );
  } catch {
    return [];
  }
}
export function readSessions(): LocalSession[] {
  return parseSessions(snapshot());
}
function write(sessions: LocalSession[]) {
  localStorage.setItem(key, JSON.stringify(sessions));
  window.dispatchEvent(new Event(eventName));
}
export function rememberSession(session: LocalSession) {
  write([session, ...readSessions().filter((item) => item.session_id !== session.session_id)]);
}
export function removePaperSessions(paper: string) {
  write(readSessions().filter((session) => session.paper_id !== paper));
}
export function forgetSession(id: string) {
  write(readSessions().filter((session) => session.session_id !== id));
}
export function useLocalSessions(paper?: string) {
  const raw = useSyncExternalStore(subscribe, snapshot, () => "[]");
  return useMemo(
    () => parseSessions(raw).filter((session) => !paper || session.paper_id === paper),
    [raw, paper],
  );
}
