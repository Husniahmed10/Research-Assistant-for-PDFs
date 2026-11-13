"""Node 10: attach only verified source/page references."""

import re

from agents.state import AgentState
from agents.nodes._shared import UNVERIFIED, skipped, update
from services.citations import build_citations


async def citation_formatter_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    if state.get("is_output_safe") is not True:
        return update("Citations withheld for unverified answer", citations=[], final_answer=UNVERIFIED)
    answer = re.sub(r"\n\nSources: Page \d+(?:, Page \d+)*$", "", state.get("final_answer", ""))
    try:
        citations = build_citations(answer, state.get("source_chunks", []))
    except (ValueError, KeyError, TypeError):
        return update("Invalid citation blocked", citations=[], final_answer=UNVERIFIED,
                      is_output_safe=False, error="Answer contains an unknown source reference")
    if citations:
        answer += "\n\nSources: " + ", ".join(f"Page {citation['page']}" for citation in citations)
    return update("Verified citations attached", citations=citations, final_answer=answer)
