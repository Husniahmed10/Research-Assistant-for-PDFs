"""
agents/nodes/citation_formatter.py — Citation Extraction & Formatting (Node 10)
=================================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import AgentState

Step 2:  Write citation_formatter_node(state: AgentState) → dict:
            - Extract state["reranked_chunks"] and state["final_answer"]

            - For each chunk in reranked_chunks:
                * Extract: page_num, a short snippet (first 100 chars of chunk text)
                * Build citation dict: { "page": page_num, "snippet": snippet, "chunk_id": id }

            - Deduplicate citations by page number

            - Append citation references to the final_answer:
                e.g., add a "Sources:" section at the bottom:
                "📚 Sources: Page 3, Page 7, Page 12"

            - Set state["citations"] = list of citation dicts
            - Set state["final_answer"] = answer + formatted citation footer
            - Append "📌 Citations attached" to state["steps"]
            - Return updated state dict

NOTE: The frontend uses state["citations"] to display highlighted source panels.
      Each citation links back to the specific page of the uploaded PDF.
"""

