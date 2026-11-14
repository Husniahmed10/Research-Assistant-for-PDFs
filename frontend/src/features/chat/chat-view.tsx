"use client";
import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  Copy,
  MessageSquare,
  Plus,
  Square,
} from "lucide-react";
import type { Citation, LocalSession, Message } from "@/lib/types";
import { useMessages } from "@/lib/queries";
import { Button, ErrorNotice, Skeleton } from "@/components/ui";
import { Markdown } from "@/components/markdown";
import { MessageEvaluation } from "@/features/quality/evaluation";
import { copyText, EvidencePanel } from "./evidence-panel";
import { useChat } from "./use-chat";

const suggestions = [
  "What are the main contributions?",
  "Explain the methodology simply",
  "What are the limitations?",
];
function AssistantMessage({
  message,
  onCitation,
}: {
  message: Message;
  onCitation: (citation: Citation) => void;
}) {
  return (
    <article className="chat-message assistant-message">
      <span className="assistant-avatar">
        <BookOpen size={17} />
      </span>
      <div className="message-body">
        <div className="message-heading">
          <strong>PaperLens</strong>
          <span>Research assistant</span>
        </div>
        <Markdown content={message.content} citations={message.citations} onCitation={onCitation} />
        {!!message.citations.length && (
          <div className="citation-chips">
            {message.citations.map((citation) => (
              <button
                key={`${citation.page_num}-${citation.id}`}
                onClick={() => onCitation(citation)}
              >
                <BookOpen size={12} />
                Page {citation.page_num}
                <span>{citation.source_ids.map((id) => `[${id}]`).join(" ")}</span>
              </button>
            ))}
          </div>
        )}
        <div className="answer-meta">
          <button
            className="icon-button"
            aria-label="Copy answer"
            onClick={() => void copyText(message.content)}
          >
            <Copy size={14} />
          </button>
          {message.model_used && message.model_used !== "none" && <span>{message.model_used}</span>}
          {message.latency_ms != null && <span>{(message.latency_ms / 1000).toFixed(1)}s</span>}
          <span>Saved</span>
        </div>
        <MessageEvaluation messageId={message.id} />
      </div>
    </article>
  );
}
export function ChatView({
  session,
  sessions,
  onSessionChange,
  onNewChat,
  creating,
  active = true,
  initialQuestion = "",
  onInitialQuestionUsed,
}: {
  session: LocalSession;
  sessions: LocalSession[];
  onSessionChange: (id: string) => void;
  onNewChat: () => void;
  creating: boolean;
  active?: boolean;
  initialQuestion?: string;
  onInitialQuestionUsed: () => void;
}) {
  const history = useMessages(session.session_id);
  const { run, send, stop, checkHistory, dismiss } = useChat(session.session_id);
  const [query, setQuery] = useState(initialQuestion);
  const [previousQuestion, setPreviousQuestion] = useState(initialQuestion);
  // A summary shortcut may change while this panel stays mounted. Sync only
  // a new non-empty shortcut; consuming it must not erase the user's draft.
  if (initialQuestion !== previousQuestion) {
    setPreviousQuestion(initialQuestion);
    if (initialQuestion) setQuery(initialQuestion);
  }
  const [citation, setCitation] = useState<Citation | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const textArea = useRef<HTMLTextAreaElement>(null);
  const busy = run.status === "streaming" || run.status === "reconciling";
  const canSend = run.status === "idle" && !history.isPending && !history.isError && !creating;
  useEffect(() => {
    if (active) end.current?.scrollIntoView({ behavior: "instant", block: "end" });
  }, [active, history.data?.length, run.answer, run.steps.length]);
  useEffect(() => {
    if (active && initialQuestion) {
      textArea.current?.focus();
      onInitialQuestionUsed();
    }
  }, [active, initialQuestion, onInitialQuestionUsed]);
  function submit() {
    if (!canSend || !query.trim() || query.length > 8000) return;
    const question = query.trim();
    setQuery("");
    void send(question);
  }
  return (
    <section className={`chat-view ${citation ? "evidence-open" : ""}`}>
      <div className="chat-toolbar">
        <label>
          <MessageSquare size={15} />
          <span className="sr-only">Conversation in this browser</span>
          <select
            value={session.session_id}
            disabled={busy || creating}
            onChange={(event) => onSessionChange(event.target.value)}
          >
            {sessions.map((item) => (
              <option key={item.session_id} value={item.session_id}>
                {item.title} · {new Date(item.created_at).toLocaleDateString()}
              </option>
            ))}
          </select>
        </label>
        <Button variant="ghost" disabled={busy || creating} onClick={onNewChat}>
          <Plus size={15} />
          New chat
        </Button>
      </div>
      <p className="local-session-note">
        Conversation links saved in this browser · Message history saved by the API
      </p>
      <div className="chat-scroll" role="log" aria-label="Conversation" aria-live="off">
        {history.isPending ? (
          <Skeleton rows={3} />
        ) : history.isError ? (
          <ErrorNotice error={history.error} retry={() => void history.refetch()} />
        ) : (
          <>
            {!history.data?.length && run.status === "idle" && (
              <div className="chat-welcome">
                <span className="welcome-mark">
                  <BookOpen size={30} strokeWidth={1.4} />
                </span>
                <div className="eyebrow">READ BETWEEN THE LINES</div>
                <h2>What would you like to understand?</h2>
                <p>
                  Explore ideas, unpack methods, and question the findings.
                  <br />
                  Your paper is the starting point.
                </p>
                <div className="suggestions">
                  {suggestions.map((question) => (
                    <button
                      key={question}
                      onClick={() => {
                        setQuery(question);
                        textArea.current?.focus();
                      }}
                    >
                      {question}
                      <ArrowUp size={14} />
                    </button>
                  ))}
                </div>
              </div>
            )}
            {history.data?.map((message) =>
              message.role === "assistant" ? (
                <AssistantMessage key={message.id} message={message} onCitation={setCitation} />
              ) : (
                <article key={message.id} className="chat-message user-message">
                  <div className="user-bubble">
                    <span className="user-label">You</span>
                    <p>{message.content}</p>
                  </div>
                </article>
              ),
            )}
            {run.status !== "idle" && (
              <>
                <article className="chat-message user-message">
                  <div className="user-bubble">
                    <span className="user-label">You</span>
                    <p>{run.query}</p>
                  </div>
                </article>
                <article className="chat-message assistant-message">
                  <span className="assistant-avatar">
                    <BookOpen size={17} />
                  </span>
                  <div className="message-body">
                    <div className="message-heading">
                      <strong>PaperLens</strong>
                      <span>
                        {run.status === "streaming"
                          ? "Working with your paper"
                          : "Checking saved history"}
                      </span>
                    </div>
                    {run.steps.length > 0 && (
                      <details className="workflow">
                        <summary>
                          {run.steps.length} workflow updates
                          <ChevronDown size={13} />
                        </summary>
                        <ol>
                          {run.steps.map((step, index) => (
                            <li key={index}>
                              <Check size={12} />
                              {step}
                            </li>
                          ))}
                        </ol>
                      </details>
                    )}
                    {run.answer ? (
                      <div className="streaming-answer">
                        <Markdown content={run.answer} />
                        <span className="small muted">
                          {run.status === "interrupted"
                            ? "Partial display · verify saved history below"
                            : "Receiving validated answer…"}
                        </span>
                      </div>
                    ) : (
                      run.status === "streaming" && (
                        <div className="thinking" role="status">
                          <span />
                          <span />
                          <span />
                          <span className="sr-only">Running paper analysis</span>
                        </div>
                      )
                    )}
                  </div>
                </article>
              </>
            )}
            {run.status === "interrupted" && (
              <div className="interrupted" role="alert">
                <strong>Connection interrupted</strong>
                <p>{run.error}</p>
                <p>
                  The backend may still be running. Refresh saved history before sending again to
                  avoid a duplicate question.
                </p>
                <div>
                  <Button variant="secondary" onClick={() => void checkHistory()}>
                    Check saved history
                  </Button>
                  {run.reconciled && (
                    <Button
                      variant="ghost"
                      onClick={() => {
                        setQuery(run.query);
                        dismiss();
                        textArea.current?.focus();
                      }}
                    >
                      Edit / explicitly retry
                    </Button>
                  )}
                </div>
              </div>
            )}
          </>
        )}
        <div ref={end} />
      </div>
      <form
        className="composer-area"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <div className="composer">
          <textarea
            ref={textArea}
            aria-label="Ask about this paper"
            placeholder="Ask a question about this paper…"
            rows={2}
            maxLength={8000}
            value={query}
            disabled={!canSend}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                submit();
              }
            }}
          />
          <div className="composer-bottom">
            <span>
              {busy
                ? run.status === "reconciling"
                  ? "Checking the saved conversation…"
                  : "Following the evidence…"
                : "Enter to send · Shift + Enter for a new line"}
            </span>
            {run.status === "streaming" ? (
              <button
                type="button"
                className="send-button stop-button"
                aria-label="Stop receiving answer"
                onClick={stop}
              >
                <Square size={14} fill="currentColor" />
              </button>
            ) : (
              <button
                className="send-button"
                type="submit"
                disabled={!canSend || !query.trim()}
                aria-label="Send question"
              >
                <ArrowUp size={20} />
              </button>
            )}
          </div>
        </div>
        <p className="composer-disclaimer">
          AI can make mistakes. Check important claims against the cited evidence.
        </p>
        <span className="sr-only" role="status" aria-live="polite">
          {run.status === "streaming"
            ? run.steps.at(-1) || "Generating answer"
            : run.status === "reconciling"
              ? "Checking saved history"
              : run.status === "interrupted"
                ? "Connection interrupted"
                : "Ready for a question"}
        </span>
      </form>
      <EvidencePanel citation={active ? citation : null} onClose={() => setCitation(null)} />
    </section>
  );
}
