"""Node 10: attach only verified source/page references."""

import re

from agents.state import AgentState
from agents.nodes._shared import UNVERIFIED, skipped, update


async def citation_formatter_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    if state.get("is_output_safe") is not True:
        return update("Citations withheld for unverified answer", citations=[], final_answer=UNVERIFIED)
    answer = re.sub(r"\n\nSources: Page \d+(?:, Page \d+)*$", "", state.get("final_answer", ""))
    sources = {chunk["source_id"]: chunk for chunk in state.get("source_chunks", [])}
    references = list(dict.fromkeys(re.findall(r"\[(\d+)\]", answer)))
    if any(reference not in sources for reference in references):
        return update("Invalid citation blocked", citations=[], final_answer=UNVERIFIED,
                      is_output_safe=False, error="Answer contains an unknown source reference")
    pages = {}
    for reference in references:
        chunk = sources[reference]
        page = int(chunk["page_num"])
        if page not in pages:
            pages[page] = {"page": page, "page_num": page, "id": reference,
                           "chunk_id": chunk["chunk_id"], "snippet": chunk["text"][:100],
                           "source_ids": [], "chunk_ids": []}
        pages[page]["source_ids"].append(reference)
        pages[page]["chunk_ids"].append(chunk["chunk_id"])
    if pages:
        answer += "\n\nSources: " + ", ".join(f"Page {page}" for page in pages)
    return update("Verified citations attached", citations=list(pages.values()), final_answer=answer)
