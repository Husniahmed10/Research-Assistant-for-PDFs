"""
agents/state.py — LangGraph Agent State Definition
====================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import TypedDict, Annotated, List from typing
Step 2:  Import operator for list reduction

Step 3:  Define AgentState(TypedDict) — this is the shared state passed between all nodes:
            Fields:
            - query: str                       → original user question
            - paper_id: str                    → which paper to search
            - session_id: str                  → chat session for history
            - chat_history: list[dict]         → previous messages (role + content)

            - is_safe: bool                    → set by input_guard node
            - rejection_reason: str            → why query was rejected (if unsafe)

            - query_type: str                  → "factual" | "summary" | "comparative" | "definition"
            - rewritten_query: str             → improved query from query_rewriter node
            - multi_queries: list[str]         → 3 query variants from multi_query node
            - hyde_document: str               → hypothetical answer from hyde_generator node

            - retrieved_chunks: list[dict]     → raw results from pinecone hybrid search
            - reranked_chunks: list[dict]      → top-N after Cohere reranking

            - final_answer: str                → generated answer from synthesizer node
            - citations: list[dict]            → page numbers + snippets used
            - model_used: str                  → "openai" or "groq"
            - is_output_safe: bool             → set by output_guard node

            - steps: list[str]                 → track which nodes ran (for UI display)
            - error: str                       → any error message

Step 4:  Export AgentState for use in graph.py and all node files
"""

