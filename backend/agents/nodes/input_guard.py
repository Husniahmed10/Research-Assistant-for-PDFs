"""Node 1: validate input through NeMo input rails."""

from agents.state import AgentState
from agents.nodes._shared import bounded, history, update
from services import guardrails_service
from services.input_policy import blocks_input
from services.conversation_memory import conversation_reply, extract_user_name, is_name_recall_query


async def input_guard_node(state: AgentState) -> AgentState:
    query = state.get("query", "").strip()
    try:
        if not query or bounded(query, 2000) != query:
            raise ValueError("Question must contain at most 2000 tokens")
        name = None
        recall = is_name_recall_query(query)
        if recall:
            name = state.get("user_name") if "user_name" in state else extract_user_name([
                item["content"] for item in history(state) if item["role"] == "user"
            ])
        answer = conversation_reply(query, name)
        if answer is not None:
            return update("Answered name recall from chat history" if recall else "Handled conversation",
                          is_safe=True, is_output_safe=True,
                          query_type="conversation", final_answer=answer, model_used="none", citations=[],
                          source_chunks=[], reranked_chunks=[], error="", rejection_reason="",
                          output_rejection_reason="", generation_attempts=0, retry_count=0)
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
                  query_type="",
                  final_answer="", citations=[], is_output_safe=False, generation_attempts=0, retry_count=0)
