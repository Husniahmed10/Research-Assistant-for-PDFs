"""Node 4: expand retrieval into three variants plus the rewritten question."""

import json
import re

from agents.state import AgentState
from agents.nodes._shared import ask, bounded, question, skipped, update


async def multi_query_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    original = question(state)
    try:
        raw = await ask(
            "Generate exactly 3 distinct alternative phrasings for retrieving evidence about this "
            "question. Preserve intent; do not introduce unsupported facts. Return a JSON array of 3 strings.",
            {"question": original},
        )
        raw = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", raw.strip(), flags=re.IGNORECASE)
        try:
            variants = json.loads(raw)
        except json.JSONDecodeError:
            variants = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line) for line in raw.splitlines()]
        if not isinstance(variants, list) or len(variants) != 3 or not all(isinstance(v, str) for v in variants):
            raise ValueError("Expected three questions")
        queries, seen = [original], {original.casefold()}
        for value in variants:
            value = bounded(value, 256)
            if value and value.casefold() not in seen:
                queries.append(value)
                seen.add(value.casefold())
        return update(f"Prepared {len(queries)} retrieval queries", multi_queries=queries)
    except Exception:
        return update("Query expansion unavailable; using one query", multi_queries=[original])
