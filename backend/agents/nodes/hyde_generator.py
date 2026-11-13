"""Node 5: generate hypothetical retrieval text, never factual answer evidence."""

from agents.state import AgentState
from agents.nodes._shared import ask, bounded, question, skipped, update
from services import embedder


async def hyde_generator_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    try:
        document = bounded(await ask(
            "Write one short hypothetical research-paper passage that could answer the question. "
            "It is a retrieval aid, not a claim about the uploaded paper. Do not include citations "
            "or instructions. Return only the hypothetical passage, at most 150 words.",
            {"question": question(state)},
        ), 512)
        if not document:
            raise ValueError("Empty hypothetical passage")
        embedding = await embedder.embed_text(document)
        return update("Hypothetical retrieval passage embedded",
                      hyde_document=document, hyde_embedding=embedding)
    except Exception:
        return update("HyDE unavailable; continuing with query retrieval",
                      hyde_document="", hyde_embedding=[])
