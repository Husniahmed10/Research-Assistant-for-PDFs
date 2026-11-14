"""Failure-isolated RAGAS scoring; call the synchronous evaluator in a worker."""

from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite
from time import perf_counter
import logging

import logfire
from starlette.concurrency import run_in_threadpool

from core.config import settings
from db.database import AsyncSessionLocal
from db import crud

_WEIGHTS = {"faithfulness": 0.40, "answer_relevancy": 0.35, "context_precision": 0.25}
logger = logging.getLogger(__name__)
_FAILURES = {
    "missing_api_key": "Answer-quality evaluation is not configured. Check the evaluation provider credentials.",
    "invalid_input": "This answer does not have the question and source excerpts needed for evaluation.",
    "dependency_error": "The answer-quality service could not start. Check the backend evaluation dependencies and restart it.",
    "authentication_error": "The evaluation provider rejected authorization. Check its API key and permissions.",
    "rate_limit": "The evaluation provider is rate-limited or its quota is exhausted. Try later or check API billing.",
    "timeout": "Answer-quality evaluation timed out. The saved answer is still available.",
    "incomplete_scores": "The evaluator could not produce all quality scores. The saved answer is still available.",
    "evaluation_error": "Answer-quality evaluation failed. Check the backend evaluation logs; the saved answer is still available.",
}


def _failure_code(error: Exception, stage: str) -> str:
    if stage == "configuration":
        return "missing_api_key"
    if stage == "validation":
        return "invalid_input"
    if stage == "dependencies":
        return "dependency_error"
    if getattr(error, "status_code", None) in {401, 403}:
        return "authentication_error"
    if getattr(error, "status_code", None) == 429:
        return "rate_limit"
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        return "timeout"
    return "evaluation_error"


def _score(value):
    if value is None or isinstance(value, (bool, str)):
        return None
    try:
        value = float(value)
        return value if isfinite(value) and 0 <= value <= 1 else None
    except (ValueError, TypeError, OverflowError):
        return None


def compute_overall(scores: dict) -> float | None:
    """Require all three valid metrics; missing scores are never treated as zero."""
    values = {name: _score(scores.get(name)) for name in _WEIGHTS}
    if any(value is None for value in values.values()):
        return None
    return round(sum(values[name] * weight for name, weight in _WEIGHTS.items()), 3)


def score_to_label(score: float | None) -> str | None:
    value = _score(score)
    if value is None:
        return None
    return "Excellent" if value >= 0.8 else "Good" if value >= 0.6 else "Fair" if value >= 0.4 else "Poor"


def evaluate_rag_response(query: str, answer: str, contexts: list[str], ground_truth: str | None = None) -> dict:
    """Return JSON-safe scores, including nulls for missing/failed measurements.

    Without a human reference, use context_utilization, RAGAS's reference-free
    context precision variant. precision_metric makes this distinction explicit.
    Imports and model construction are inside the failure boundary so an optional
    evaluation dependency or provider failure never prevents chat startup.
    """
    started = perf_counter()
    result = {name: None for name in _WEIGHTS}
    result.update(context_recall=None, overall_score=None, label=None, status="failed",
                  precision_metric="context_precision" if ground_truth is not None else "context_utilization",
                  error_code=None, error_message=None)
    stage = "configuration"
    try:
        if not settings.OPENAI_API_KEY:
            raise ValueError("Evaluation provider is not configured")
        stage = "validation"
        if not isinstance(query, str) or not query.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError("Question and answer are required")
        if not isinstance(contexts, list) or not contexts or any(not isinstance(c, str) or not c.strip() for c in contexts):
            raise ValueError("Non-empty source contexts are required")
        if ground_truth is not None and (not isinstance(ground_truth, str) or not ground_truth.strip()):
            raise ValueError("Reference answer must be non-empty")
        stage = "dependencies"
        import ragas
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_utilization, context_recall
        from ragas.run_config import RunConfig

        data = {"question": [query], "answer": [answer], "contexts": [contexts]}
        precision = context_precision if ground_truth is not None else context_utilization
        metrics = [faithfulness, answer_relevancy, precision]
        if ground_truth is not None:
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)
        stage = "evaluation"
        with logfire.span("ragas.evaluate", model="gpt-4o-mini", context_count=len(contexts)):
            evaluated = ragas.evaluate(
                dataset=Dataset.from_dict(data), metrics=deepcopy(metrics),
                llm=ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, timeout=45, max_retries=1),
                embeddings=OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY,
                                            request_timeout=45, max_retries=1),
                run_config=RunConfig(timeout=60, max_retries=1, max_workers=2),
                # Catch provider failures here rather than losing their safe
                # classification when RAGAS replaces every failed metric by NaN.
                raise_exceptions=True,
            )
            row = evaluated.to_pandas().iloc[0].to_dict()
            raw = {"faithfulness": _score(row.get("faithfulness")),
                   "answer_relevancy": _score(row.get("answer_relevancy")),
                   "context_precision": _score(row.get(result["precision_metric"]))}
            result.update({key: round(value, 3) if value is not None else None for key, value in raw.items()})
            if ground_truth is not None:
                recall = _score(row.get("context_recall"))
                result["context_recall"] = round(recall, 3) if recall is not None else None
            result["overall_score"] = compute_overall(raw)
            result["label"] = score_to_label(result["overall_score"])
            result["status"] = "completed" if result["overall_score"] is not None else "failed"
            if result["status"] == "failed":
                result.update(error_code="incomplete_scores", error_message=_FAILURES["incomplete_scores"])
                logger.warning("RAG evaluation returned incomplete scores")
    except Exception as exc:
        # Do not expose provider messages, credentials, prompts or source excerpts.
        code = _failure_code(exc, stage)
        result.update(status="failed", error_code=code, error_message=_FAILURES[code])
        logger.warning("RAG evaluation failed: %s (%s, stage=%s)", code, type(exc).__name__, stage)
        logfire.warn("RAG evaluation failed", error_code=code, error_type=type(exc).__name__, stage=stage)
    result["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    logfire.info("RAG evaluation finished", status=result["status"], overall_score=result["overall_score"],
                 latency_ms=(perf_counter() - started) * 1000)
    return result


async def evaluate_message(message_id: str, query: str, answer: str, contexts: list[str]) -> None:
    """Background task: use a fresh DB session, never the request's session."""
    try:
        scores = await run_in_threadpool(evaluate_rag_response, query, answer, contexts)
        async with AsyncSessionLocal() as db:
            await crud.save_ragas_scores(db, message_id, scores)
            await db.commit()
    except Exception:
        logfire.warn("Background evaluation unavailable")
        # Best effort: mark failed even when a dependency or the worker failed.
        try:
            async with AsyncSessionLocal() as db:
                await crud.save_ragas_scores(db, message_id, {"status": "failed"})
                await db.commit()
        except Exception:
            logfire.warn("Could not persist evaluation status")
