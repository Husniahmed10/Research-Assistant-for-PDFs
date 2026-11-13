"""Node 3: resolve follow-ups and improve retrieval wording."""

from agents.state import AgentState
from agents.nodes._shared import ask, bounded, history, skipped, update


async def query_rewriter_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    try:
        rewritten = bounded(await ask(
            "Rewrite the question as one standalone research-paper retrieval question. Resolve references "
            "using the conversation when possible. Preserve the user's intent and constraints. Do not "
            "invent acronyms, datasets, methods, results, or answers. Return only the question.",
            {"question": state["query"], "type": state.get("query_type", "factual"), "history": history(state)},
        ), 256)
        if not rewritten:
            raise ValueError("Empty rewrite")
        return update("Question rewritten for retrieval", rewritten_query=rewritten)
    except Exception:
        return update("Rewrite unavailable; using original question", rewritten_query=state["query"])
