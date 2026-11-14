"use client";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, LoaderCircle, ShieldCheck } from "lucide-react";
import { api } from "@/lib/api";
import { terminalEvaluation } from "@/lib/conversation";
import type { Evaluation, QualityReport } from "@/lib/types";
import { Button, Empty, ErrorNotice, Skeleton } from "@/components/ui";
const percentage = (value: number | null | undefined) =>
  value == null ? "Unavailable" : `${Math.round(value * 100)}%`;
export function Metric({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="metric">
      <div>
        <span>{label}</span>
        <strong>{percentage(value)}</strong>
      </div>
      <div className="metric-track" aria-hidden="true">
        {value != null && <span style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />}
      </div>
    </div>
  );
}
function EvaluationDetails({ value }: { value: Evaluation }) {
  return (
    <>
      <Metric label="Faithfulness" value={value.faithfulness} />
      <Metric label="Answer relevancy" value={value.answer_relevancy} />
      <Metric
        label={
          value.precision_metric === "context_utilization"
            ? "Context utilization · reference-free"
            : "Context precision"
        }
        value={value.context_precision}
      />
      <p className="evaluation-note">
        Automated estimates, not factual guarantees. Review the cited evidence.
      </p>
    </>
  );
}
export function MessageEvaluation({ messageId }: { messageId: string }) {
  const [expanded, setExpanded] = useState(false);
  const [started, setStarted] = useState(() => Date.now());
  const [timedOut, setTimedOut] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setTimedOut(true), 90000);
    return () => clearTimeout(timer);
  }, [started]);
  const evaluation = useQuery({
    queryKey: ["evaluation", messageId],
    queryFn: ({ signal }) => api.evaluation(messageId, signal),
    refetchInterval: (query) =>
      !terminalEvaluation(query.state.data?.status) && Date.now() - started < 90000 ? 3500 : false,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const value = evaluation.data;
  const pending = !value || value.status === "pending";
  const expired = pending && timedOut;
  const labels = {
    failed: "Evaluation unavailable",
    skipped: "Evaluation skipped",
    not_scheduled: "Evaluation not scheduled",
    pending: "Evaluating answer…",
    completed: `${value?.label || "Evaluated"} · ${percentage(value?.overall_score)}`,
  };
  return (
    <div className="message-evaluation">
      <button
        className="evaluation-toggle"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        {pending && !expired ? (
          <LoaderCircle size={13} className="spin" />
        ) : (
          <ShieldCheck size={14} />
        )}
        <span>
          {evaluation.isError
            ? "Evaluation unavailable"
            : expired
              ? "Evaluation taking longer than usual"
              : value
                ? labels[value.status]
                : "Checking evaluation…"}
        </span>
        <ChevronDown size={12} className={expanded ? "rotate" : ""} />
      </button>
      {expanded && (
        <div className="evaluation-details">
          {evaluation.isError ? (
            <ErrorNotice
              error={evaluation.error}
              retry={() => {
                setTimedOut(false);
                setStarted(Date.now());
                void evaluation.refetch();
              }}
            />
          ) : value?.status === "completed" ? (
            <EvaluationDetails value={value} />
          ) : (
            <>
              <p className="small muted">
                {value?.status === "skipped" || value?.status === "not_scheduled"
                  ? "Conversational responses and answers without supporting excerpts may not be evaluated."
                  : value?.status === "failed"
                    ? value.error_message ||
                      "The evaluation provider could not score this answer. The saved conversation is still available."
                    : "Quality evaluation runs after the answer has been saved."}
              </p>
              {pending && (
                <Button
                  variant="ghost"
                  onClick={() => {
                    setTimedOut(false);
                    setStarted(Date.now());
                    void evaluation.refetch();
                  }}
                >
                  Check again
                </Button>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
function Report({ data }: { data: QualityReport }) {
  const utilization = !!data.precision_metrics.context_utilization;
  const mixed = utilization && !!data.precision_metrics.context_precision;
  return (
    <>
      <div className="quality-overview">
        <span className="quality-emblem">
          <ShieldCheck size={31} strokeWidth={1.5} />
        </span>
        <div>
          <div className="eyebrow">OVERALL QUALITY</div>
          <h2>{percentage(data.avg_overall)}</h2>
          <p>{data.label || "No overall score available"}</p>
        </div>
        <div className="quality-count">
          <strong>{data.evaluated}</strong>
          <span>evaluated answers</span>
        </div>
      </div>
      <div className="report-metrics">
        <Metric label="Faithfulness" value={data.avg_faithfulness} />
        <Metric label="Answer relevancy" value={data.avg_relevancy} />
        <Metric
          label={
            mixed
              ? "Context precision / utilization · mixed"
              : utilization
                ? "Context utilization · reference-free"
                : "Context precision"
          }
          value={data.avg_precision}
        />
      </div>
      <div className="report-counts">
        <span>
          {data.total_answers} {data.total_answers === 1 ? "answer" : "answers"}
        </span>
        <span>{data.pending} pending</span>
        <span>{data.skipped + data.not_scheduled} skipped / not scheduled</span>
        <span>{data.failed} unavailable</span>
      </div>
      {data.model_usage && Object.keys(data.model_usage).length > 0 && (
        <div className="model-usage">
          <h3>Models used</h3>
          {Object.entries(data.model_usage).map(([model, count]) => (
            <div key={model}>
              <span>{model === "none" ? "Conversation / policy response" : model}</span>
              <span>
                {count} {count === 1 ? "answer" : "answers"}
              </span>
            </div>
          ))}
        </div>
      )}
      <p className="evaluation-note">
        Scores are automated estimates, not factual guarantees. Reference-free context utilization
        measures how the answer uses its retrieved context; it does not establish that all relevant
        evidence was retrieved. Missing scores are shown as unavailable.
      </p>
    </>
  );
}
export function QualityView({ paperId, sessionId }: { paperId: string; sessionId?: string }) {
  const [scope, setScope] = useState<"papers" | "sessions">("papers");
  const id = scope === "papers" ? paperId : sessionId;
  const [started, setStarted] = useState(() => Date.now());
  const report = useQuery({
    queryKey: ["quality", scope, id],
    queryFn: ({ signal }) => api.report(scope, id!, signal),
    enabled: !!id,
    refetchInterval: (query) =>
      query.state.data?.pending && Date.now() - started < 90000 ? 5000 : false,
  });
  return (
    <section className="quality-view">
      <div className="view-heading">
        <div className="eyebrow">A CLOSER LOOK AT ANSWER QUALITY</div>
        <h2>Research you can examine.</h2>
        <p className="muted">Review how well answers reflect the retrieved evidence.</p>
      </div>
      <div className="quality-controls">
        <div className="segmented">
          <button className={scope === "papers" ? "active" : ""} onClick={() => setScope("papers")}>
            This paper
          </button>
          <button
            disabled={!sessionId}
            className={scope === "sessions" ? "active" : ""}
            onClick={() => setScope("sessions")}
          >
            Current conversation
          </button>
        </div>
        <Button
          variant="ghost"
          busy={report.isFetching}
          onClick={() => {
            setStarted(Date.now());
            void report.refetch();
          }}
        >
          Refresh
        </Button>
      </div>
      {report.isPending ? (
        <Skeleton rows={3} />
      ) : report.isError ? (
        <ErrorNotice error={report.error} retry={() => void report.refetch()} />
      ) : report.data && report.data.total_answers ? (
        <div className="report-card">
          <Report data={report.data} />
        </div>
      ) : (
        <Empty title="Quality follows the conversation">
          Ask a question about this paper. Evaluations will appear after supported answers are
          saved.
        </Empty>
      )}
    </section>
  );
}
