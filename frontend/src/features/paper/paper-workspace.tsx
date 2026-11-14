"use client";
import { useCallback, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  ArrowUpRight,
  BookOpen,
  ChartNoAxesCombined,
  FileText,
  LoaderCircle,
  MessageSquare,
  Plus,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { api, errorMessage } from "@/lib/api";
import { usePapers } from "@/lib/queries";
import { rememberSession, useLocalSessions } from "@/lib/sessions";
import { summarySections, type LocalSession } from "@/lib/types";
import { Button, Empty, ErrorNotice, Skeleton, Status } from "@/components/ui";
import { Markdown } from "@/components/markdown";
import { DeletePaper } from "@/features/library/delete-paper";
import { ChatView } from "@/features/chat/chat-view";
import { QualityView } from "@/features/quality/evaluation";
export function PaperWorkspace({ paperId }: { paperId: string }) {
  const router = useRouter();
  const papers = usePapers();
  const paper = papers.data?.find((item) => item.id === paperId);
  const [tab, setCurrentTab] = useState("summary");
  const [chatVisited, setChatVisited] = useState(false);
  const [activeSession, setActiveSession] = useState<string>();
  const [ephemeral, setEphemeral] = useState<LocalSession[]>([]);
  const sessions = useLocalSessions(paperId);
  const allSessions = [
    ...ephemeral.filter((session) => session.paper_id === paperId),
    ...sessions.filter((saved) => !ephemeral.some((item) => item.session_id === saved.session_id)),
  ];
  const selectedSession =
    allSessions.find((item) => item.session_id === activeSession) || allSessions[0];
  const [question, setQuestion] = useState("");
  const clearQuestion = useCallback(() => setQuestion(""), []);
  const summary = useQuery({
    queryKey: ["summary", paperId],
    queryFn: ({ signal }) => api.summary(paperId, signal),
    enabled: paper?.status === "ready",
  });
  const create = useMutation({
    mutationFn: () => api.createSession(paperId),
    onSuccess: (session) => {
      const local = { ...session, title: `Conversation ${allSessions.length + 1}` };
      setEphemeral((items) => [local, ...items]);
      setActiveSession(session.session_id);
      try {
        rememberSession(local);
      } catch {
        toast.error(
          "Browser storage is unavailable. This chat link will last for this visit only.",
        );
      }
    },
    onError: (error) => toast.error(errorMessage(error)),
  });
  function setTab(value: string) {
    setCurrentTab(value);
    if (value === "chat") setChatVisited(true);
  }
  function ask(question: string) {
    setQuestion(question);
    setTab("chat");
  }
  function selectSession(id: string) {
    setActiveSession(id);
    const session = allSessions.find((item) => item.session_id === id);
    if (session) {
      try {
        rememberSession(session);
      } catch {
        toast.error("The selected conversation could not be remembered in this browser.");
      }
    }
  }
  if (papers.isPending)
    return (
      <div className="page">
        <Skeleton rows={4} />
      </div>
    );
  if (papers.isError)
    return (
      <div className="page">
        <ErrorNotice error={papers.error} retry={() => void papers.refetch()} />
      </div>
    );
  if (!paper)
    return (
      <div className="page">
        <Empty
          title="This paper is unavailable"
          action={
            <Link href="/" className="button primary">
              Back to library
            </Link>
          }
        >
          It may have been deleted, or this link may belong to another backend.
        </Empty>
      </div>
    );
  return (
    <div className="page workspace-page">
      <Link href="/" className="back-link">
        <ArrowLeft size={14} />
        Back to library
      </Link>
      <div className="paper-heading">
        <span className="workspace-paper-icon">
          <FileText size={28} strokeWidth={1.5} />
        </span>
        <div className="paper-heading-text">
          <div className="eyebrow">PAPER WORKSPACE</div>
          <h1>{paper.filename}</h1>
          <p>
            {paper.page_count == null ? "Page count pending" : `${paper.page_count} pages`}
            <span>·</span>
            {(paper.file_size_kb / 1024).toFixed(2)} MB<span>·</span>Added{" "}
            {new Date(paper.created_at).toLocaleDateString()}
          </p>
        </div>
        <Status status={paper.status} />
        <DeletePaper paper={paper} onDeleted={() => router.push("/")}>
          <Trash2 size={17} />
        </DeletePaper>
      </div>
      {paper.status !== "ready" ? (
        <div className="processing-state">
          {paper.status === "failed" ? (
            <Empty title="We couldn’t prepare this paper">
              Delete this paper and upload it again. If the problem continues, check the backend
              processing logs.
            </Empty>
          ) : (
            <>
              <span className="processing-icon">
                <LoaderCircle size={29} className="spin" />
              </span>
              <h2>Getting your paper ready</h2>
              <p>Extracting text, creating a summary, and preparing searchable evidence.</p>
              <p className="muted small">
                This page updates automatically. You can return to the library while we work.
              </p>
              <div className="indeterminate" />
            </>
          )}
        </div>
      ) : (
        <>
          <div className="workspace-tabs" role="tablist" aria-label="Paper views">
            {[
              ["summary", FileText, "Summary"],
              ["chat", MessageSquare, "Chat with paper"],
              ["quality", ChartNoAxesCombined, "Answer quality"],
            ].map(([id, Icon, label]) => {
              const ViewIcon = Icon as typeof FileText;
              return (
                <button
                  key={id as string}
                  role="tab"
                  id={`tab-${id}`}
                  aria-controls={`panel-${id}`}
                  aria-selected={tab === id}
                  tabIndex={tab === id ? 0 : -1}
                  onClick={() => setTab(id as string)}
                  onKeyDown={(event) => {
                    const tabs = ["summary", "chat", "quality"];
                    const index = tabs.indexOf(tab);
                    const next =
                      event.key === "ArrowRight"
                        ? tabs[(index + 1) % tabs.length]
                        : event.key === "ArrowLeft"
                          ? tabs[(index + 2) % tabs.length]
                          : event.key === "Home"
                            ? tabs[0]
                            : event.key === "End"
                              ? tabs[2]
                              : null;
                    if (next) {
                      event.preventDefault();
                      setTab(next);
                      document.getElementById(`tab-${next}`)?.focus();
                    }
                  }}
                >
                  <ViewIcon size={16} />
                  {label as string}
                </button>
              );
            })}
          </div>
          <div className="workspace-panel">
            <div
              role="tabpanel"
              id="panel-summary"
              aria-labelledby="tab-summary"
              hidden={tab !== "summary"}
            >
              {tab === "summary" && (
                <section className="summary-view">
                  <div className="summary-main">
                    <div className="summary-caption">
                      <BookOpen size={15} />
                      THE ESSENTIALS, IN ONE PLACE
                    </div>
                    {summary.isPending ? (
                      <Skeleton rows={6} />
                    ) : summary.isError ? (
                      <ErrorNotice error={summary.error} retry={() => void summary.refetch()} />
                    ) : (
                      summary.data &&
                      summarySections.map(([key, title], index) => (
                        <section className={`summary-section section-${key}`} key={key}>
                          <div className="summary-number">{String(index + 1).padStart(2, "0")}</div>
                          <div>
                            <h2>{title}</h2>
                            <Markdown content={summary.data[key]} />
                          </div>
                        </section>
                      ))
                    )}
                  </div>
                  <aside className="summary-aside">
                    <div className="aside-card">
                      <span className="aside-icon">
                        <MessageSquare size={21} />
                      </span>
                      <h3>Go beyond the summary.</h3>
                      <p>Ask follow-up questions and explore the evidence behind the ideas.</p>
                      <Button onClick={() => ask("")}>
                        Start a conversation
                        <ArrowUpRight size={15} />
                      </Button>
                    </div>
                    <div className="question-starters">
                      <span className="eyebrow">KEEP EXPLORING</span>
                      {[
                        "Explain the methodology",
                        "What are the limitations?",
                        "How convincing are the results?",
                      ].map((q) => (
                        <button key={q} onClick={() => ask(q)}>
                          {q}
                          <ArrowUpRight size={14} />
                        </button>
                      ))}
                    </div>
                  </aside>
                </section>
              )}
            </div>
            <div role="tabpanel" id="panel-chat" aria-labelledby="tab-chat" hidden={tab !== "chat"}>
              {/* Hiding the panel preserves its request and draft. Unmount only
                when leaving the paper or selecting a different conversation. */}
              {chatVisited &&
                (selectedSession ? (
                  <ChatView
                    key={selectedSession.session_id}
                    session={selectedSession}
                    sessions={allSessions}
                    onSessionChange={selectSession}
                    onNewChat={() => {
                      if (!create.isPending) create.mutate();
                    }}
                    creating={create.isPending}
                    active={tab === "chat"}
                    initialQuestion={question}
                    onInitialQuestionUsed={clearQuestion}
                  />
                ) : (
                  <div className="start-conversation">
                    <Empty
                      title="A conversation with your paper"
                      action={
                        <Button busy={create.isPending} onClick={() => create.mutate()}>
                          <Plus size={16} />
                          Create conversation
                        </Button>
                      }
                    >
                      Ask questions grounded in this PDF. Your conversation will be saved so you can
                      come back to it.
                    </Empty>
                  </div>
                ))}
            </div>
            <div
              role="tabpanel"
              id="panel-quality"
              aria-labelledby="tab-quality"
              hidden={tab !== "quality"}
            >
              {tab === "quality" && (
                <QualityView paperId={paperId} sessionId={selectedSession?.session_id} />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
