"""Read persisted RAGAS scores; GET requests never invoke a paid evaluator."""

from collections import Counter
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.dependencies import get_db
from core.exceptions import ErrorHandlingRoute
from db import crud
from db.models import ChatSession, Message, MessageEvaluation
from services.evaluator import compute_overall, score_to_label

router = APIRouter(prefix="/api/v1/evaluation", tags=["Evaluation"], route_class=ErrorHandlingRoute)


def _score_payload(evaluation: MessageEvaluation) -> dict:
    scores = {"faithfulness": None, "answer_relevancy": None, "context_precision": None,
              "context_recall": None, "overall_score": None, "label": None,
              "precision_metric": None, "evaluated_at": None,
              "error_code": None, "error_message": None}
    scores.update(evaluation.scores or {})
    return {**scores, "message_id": evaluation.message_id, "status": evaluation.status}


def _aggregate(messages: list[Message]) -> dict:
    answers = [message for message in messages if message.role == "assistant"]
    completed = [message for message in answers if message.evaluation is not None
                 and message.evaluation.status == "completed" and message.evaluation.scores]
    def average(key):
        values = [message.evaluation.scores[key] for message in completed
                  if message.evaluation.scores.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    means = {name: average(name) for name in ("faithfulness", "answer_relevancy", "context_precision")}
    overall = compute_overall(means)
    statuses = Counter(message.evaluation.status if message.evaluation else "not_scheduled" for message in answers)
    return {
        "total_messages": len(messages), "total_answers": len(answers), "evaluated": len(completed),
        "pending": statuses["pending"], "failed": statuses["failed"], "skipped": statuses["skipped"],
        "not_scheduled": statuses["not_scheduled"],
        "avg_faithfulness": means["faithfulness"], "avg_relevancy": means["answer_relevancy"],
        "avg_precision": means["context_precision"], "avg_overall": overall,
        "label": score_to_label(overall),
        "precision_metrics": dict(Counter(message.evaluation.scores.get("precision_metric", "unknown")
                                           for message in completed)),
        "score_history": [{"message_id": message.id,
                           "overall_score": message.evaluation.scores["overall_score"],
                           "created_at": message.created_at.isoformat()} for message in completed],
    }


@router.get("/messages/{message_id}")
async def get_message_evaluation(message_id: UUID, db: AsyncSession = Depends(get_db)):
    message = await db.get(Message, str(message_id))
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Only assistant answers can be evaluated")
    if message.evaluation is None:
        raise HTTPException(status_code=404, detail="Evaluation was not scheduled for this message")
    payload = _score_payload(message.evaluation)
    if message.evaluation.status == "pending":
        return JSONResponse(status_code=202, content=payload)
    return payload


@router.get("/sessions/{session_id}/summary")
async def session_summary(session_id: UUID, db: AsyncSession = Depends(get_db)) -> dict:
    session_id = str(session_id)
    if await crud.get_session(db, session_id) is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"session_id": session_id, **_aggregate(await crud.get_messages_for_session(db, session_id))}


@router.get("/papers/{paper_id}/report")
async def paper_report(paper_id: UUID, db: AsyncSession = Depends(get_db)) -> dict:
    paper_id = str(paper_id)
    if await crud.get_paper(db, paper_id) is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    messages = list((await db.scalars(select(Message).join(ChatSession)
        .where(ChatSession.paper_id == paper_id).order_by(Message.created_at, Message.id))).all())
    answers = [message for message in messages if message.role == "assistant"]
    latencies = [message.latency_ms for message in answers if message.latency_ms is not None]
    report = _aggregate(messages)
    return {"paper_id": paper_id, **report, "quality_label": report["label"],
            "total_questions": sum(message.role == "user" for message in messages),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            # Successful answer model counts, not provider attempts/fallback counters.
            "model_usage": dict(Counter(message.model_used or "none" for message in answers))}
