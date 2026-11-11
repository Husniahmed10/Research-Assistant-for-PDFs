"""Node 1: validate input through NeMo input rails."""

from agents.state import AgentState
from agents.nodes._shared import bounded, history, update
from services import guardrails_service
from services.input_policy import blocks_input


async def input_guard_node(state: AgentState) -> AgentState:
    query = state.get("query", "").strip()
    try:
        if not query or bounded(query, 2000) != query:
            raise ValueError("Question must contain at most 2000 tokens")
        allowed = False if blocks_input(query) else await guardrails_service.check_input(query, history(state))
    except Exception:
        return update("Input check unavailable; request stopped", is_safe=False,
                      rejection_reason="Input validation unavailable", error="Input validation unavailable",
                      final_answer="I couldn't validate this request. Please try again.", citations=[],
                      is_output_safe=False, generation_attempts=0, retry_count=0)
    if not allowed:
        return update("Input blocked", is_safe=False, rejection_reason="Question did not pass input checks",
                      final_answer="I can only help with questions about the uploaded research paper.",
                      citations=[], error="", is_output_safe=False, generation_attempts=0, retry_count=0)
    return update("Input safety check passed", is_safe=True, rejection_reason="", error="",
                  final_answer="", citations=[], is_output_safe=False, generation_attempts=0, retry_count=0)
