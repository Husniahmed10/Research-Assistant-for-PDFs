"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, errorMessage } from "@/lib/api";
import { hasSavedAnswer } from "@/lib/conversation";
import { streamAnswer } from "@/lib/sse";
import type { Message } from "@/lib/types";

export interface ChatRun {
  status: "idle" | "streaming" | "reconciling" | "interrupted";
  query: string;
  answer: string;
  steps: string[];
  error: string;
  reconciled: boolean;
}
const idle: ChatRun = {
  status: "idle",
  query: "",
  answer: "",
  steps: [],
  error: "",
  reconciled: true,
};
// Keep unresolved requests across in-app navigation without persisting chat text.
const unresolved = new Map<string, { query: string; previousIds: Set<string> }>();
export function useChat(session: string) {
  const client = useQueryClient();
  const [run, setRun] = useState<ChatRun>(() =>
    unresolved.has(session)
      ? {
          ...idle,
          query: unresolved.get(session)!.query,
          status: "interrupted",
          reconciled: false,
          error: "A previous stream was interrupted. Check saved history before sending again.",
        }
      : idle,
  );
  const alive = useRef(true);
  const locked = useRef(false);
  const abort = useRef<AbortController | null>(null);
  const baseline = useRef(unresolved.get(session)?.previousIds || new Set<string>());
  const pendingQuery = useRef(unresolved.get(session)?.query || "");
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      abort.current?.abort();
    };
  }, []);

  const reconcile = useCallback(
    async (issue: string) => {
      if (alive.current) setRun((value) => ({ ...value, status: "reconciling", error: issue }));
      try {
        const messages = await api.messages(session, AbortSignal.timeout(15000));
        client.setQueryData(["messages", session], messages);
        if (!alive.current) return;
        if (hasSavedAnswer(messages, baseline.current, pendingQuery.current)) {
          unresolved.delete(session);
          setRun(idle);
          void client.invalidateQueries({ queryKey: ["quality"] });
        } else
          setRun((value) => ({ ...value, status: "interrupted", reconciled: true, error: issue }));
      } catch (error) {
        if (alive.current)
          setRun((value) => ({
            ...value,
            status: "interrupted",
            reconciled: false,
            error: `${issue} Saved history could not be checked: ${errorMessage(error)}`,
          }));
      }
    },
    [client, session],
  );

  const send = useCallback(
    async (query: string) => {
      query = query.trim();
      if (locked.current || !query || query.length > 8000) return;
      locked.current = true;
      pendingQuery.current = query;
      const controller = new AbortController();
      abort.current = controller;
      let streamedText = "";
      let submitted = false;
      setRun({ ...idle, query, status: "streaming", reconciled: false });
      try {
        // Establish authoritative history before a mutating GET. Failure here never submits.
        const before = await api.messages(session, controller.signal);
        baseline.current = new Set(before.map((message) => message.id));
        client.setQueryData(["messages", session], before);
        submitted = true;
        unresolved.set(session, { query, previousIds: baseline.current });
        await streamAnswer(session, query, controller.signal, (event) => {
          if (!alive.current) return;
          if (event.type === "step")
            setRun((value) => ({ ...value, steps: [...value.steps, event.content] }));
          if (event.type === "token") {
            streamedText += event.content;
            setRun((value) => ({ ...value, answer: streamedText }));
          }
          if (event.type === "done") {
            unresolved.delete(session);
            // Preserve confirmed answer even if the subsequent history read is unavailable.
            const answer = client.getQueryData<Message[]>(["messages", session]) || [];
            if (event.message_id && !answer.some((message) => message.id === event.message_id)) {
              const now = new Date().toISOString();
              client.setQueryData(
                ["messages", session],
                [
                  ...answer,
                  {
                    id: `local-${event.message_id}`,
                    role: "user",
                    content: query,
                    citations: [],
                    model_used: null,
                    latency_ms: null,
                    created_at: now,
                  },
                  {
                    id: event.message_id,
                    role: "assistant",
                    content: streamedText,
                    citations: event.citations,
                    model_used: event.model_used,
                    latency_ms: event.latency_ms,
                    created_at: now,
                  },
                ],
              );
            }
          }
        });
        if (alive.current) {
          setRun(idle);
          await client.invalidateQueries({ queryKey: ["messages", session] });
          void client.invalidateQueries({ queryKey: ["quality"] });
        }
      } catch (error) {
        if (alive.current) {
          if (submitted)
            await reconcile(
              controller.signal.aborted
                ? "The stream was stopped. The backend may still finish saving the answer."
                : errorMessage(error),
            );
          else
            setRun((value) => ({
              ...value,
              status: "interrupted",
              reconciled: true,
              error: `The question was not submitted: ${errorMessage(error)}`,
            }));
        }
      } finally {
        locked.current = false;
        abort.current = null;
      }
    },
    [client, reconcile, session],
  );
  return {
    run,
    send,
    stop: () => abort.current?.abort(),
    checkHistory: async () => {
      if (locked.current) return;
      locked.current = true;
      try {
        await reconcile(
          "The connection was interrupted. Check the saved conversation before retrying.",
        );
      } finally {
        locked.current = false;
      }
    },
    dismiss: () => {
      unresolved.delete(session);
      setRun(idle);
    },
  };
}
