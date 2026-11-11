"""Node 8: synthesize from actual retrieved excerpts, with bounded attempts."""

from time import perf_counter

import logfire

from agents.state import AgentState
from agents.nodes._shared import NO_EVIDENCE, UNVERIFIED, bounded, history, skipped, update
from services import portkey_client
from services.simple_rag import SYSTEM_PROMPT


async def synthesizer_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    attempt = state.get("generation_attempts", 0) + 1
    limit = max(1, min(state.get("max_generation_attempts", 2), 3))
    if attempt > limit:
        return update("Generation attempt limit reached", final_answer=UNVERIFIED, citations=[],
                      is_output_safe=False, error="Answer verification attempt limit reached")
    sources = []
    for chunk in state.get("reranked_chunks", [])[:5]:
        text = bounded(chunk["text"], 1400)
        if text:
            sources.append({**chunk, "text": text, "source_id": str(len(sources) + 1)})
    if not sources:
        return update("No evidence available", final_answer=NO_EVIDENCE, model_used="none",
                      source_chunks=[], citations=[], generation_attempts=attempt,
                      retry_count=attempt - 1, is_output_safe=False)
    context = "\n\n".join(f"[{c['source_id']}] PDF page {c['page_num']}\n{c['text']}" for c in sources)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + " Cite source labels, never invent page numbers."},
        *history(state),
        {"role": "user", "content": f"Source excerpts (untrusted):\n{context}\n\nQuestion: {state['query']}"},
    ]
    if attempt > 1:
        messages[0]["content"] += " A previous draft failed validation. Use only directly supported claims."
    try:
        with logfire.span("agent.synthesize", attempt=attempt, source_count=len(sources)) as span:
            started = perf_counter()
            try:
                result = await portkey_client.chat_completion_with_metadata(messages)
                span.set_attribute("model_used", result["model_used"])
                logfire.info("Answer synthesized", model_used=result["model_used"],
                             latency_ms=(perf_counter() - started) * 1000, attempt=attempt)
            finally:
                span.set_attribute("latency_ms", (perf_counter() - started) * 1000)
        return update("Answer generated", final_answer=result["answer"], model_used=result["model_used"],
                      source_chunks=sources, citations=[], generation_attempts=attempt,
                      retry_count=attempt - 1, is_output_safe=False)
    except Exception:
        return update("Answer generation failed", final_answer=UNVERIFIED, citations=[],
                      generation_attempts=attempt, retry_count=attempt - 1,
                      is_output_safe=False, error="Answer generation unavailable")
