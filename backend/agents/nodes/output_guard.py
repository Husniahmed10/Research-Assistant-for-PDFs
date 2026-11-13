"""Node 9: verify source references and run NeMo output rails."""

import re

from agents.state import AgentState
from agents.nodes._shared import UNVERIFIED, skipped, update
from services import guardrails_service


async def output_guard_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    answer = state.get("final_answer", "")
    sources = state.get("source_chunks", [])
    valid_ids = {chunk["source_id"] for chunk in sources}
    references_valid = set(re.findall(r"\[(\d+)\]", answer)) <= valid_ids
    pages = {int(chunk["page_num"]) for chunk in sources}
    mentioned_pages = {int(page) for page in re.findall(r"\b[Pp]age\s+(\d+)", answer)}
    try:
        allowed = bool(answer) and references_valid and mentioned_pages <= pages
        if allowed:
            allowed = await guardrails_service.check_output(state["query"], answer, sources)
    except Exception:
        return update("Output check unavailable; answer withheld", is_output_safe=False,
                      final_answer=UNVERIFIED, citations=[], error="Output validation unavailable")
    if allowed:
        return update("Output safety and grounding check passed", is_output_safe=True,
                      output_rejection_reason="")
    limit = max(1, min(state.get("max_generation_attempts", 2), 3))
    exhausted = state.get("generation_attempts", 0) >= limit
    return update("Output blocked; answer withheld", is_output_safe=False,
                  output_rejection_reason="Answer failed source or safety checks", final_answer=UNVERIFIED,
                  citations=[], error="Answer verification attempt limit reached" if exhausted else "")
