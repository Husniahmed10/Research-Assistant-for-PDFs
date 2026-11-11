"""
agents/nodes/query_rewriter.py — Query Rewriting for Better Retrieval (Node 3)
===============================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import portkey_client and AgentState

Step 2:  Write query_rewriter_node(state: AgentState) → dict:
            - Extract state["query"] and state["query_type"]
            - Build a rewriting prompt:
                "You are a retrieval optimization assistant.
                 Rewrite the following question to improve semantic search over a research paper.
                 - Expand abbreviations
                 - Add technical synonyms
                 - Make it more specific
                 Original question: {query}
                 Rewritten question (one line only):"
            - Call portkey_client.chat_completion()
            - Set state["rewritten_query"] = rewritten version
            - Append "✏️ Query rewritten for better retrieval" to state["steps"]
            - Return updated state dict

EXAMPLE:
    Input:  "What accuracy did they get?"
    Output: "What is the classification accuracy, F1 score, or performance metric achieved by the proposed model on benchmark datasets?"
"""

