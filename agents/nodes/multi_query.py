"""
agents/nodes/multi_query.py — Multi-Query Generation (Node 4)
==============================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import portkey_client and AgentState

Step 2:  Write multi_query_node(state: AgentState) → dict:
            - Extract state["rewritten_query"]
            - Build a multi-query prompt:
                "Generate 3 different phrasings of this research paper question
                 to improve retrieval coverage. Each phrasing should approach
                 the question from a different angle.
                 Question: {rewritten_query}
                 Return exactly 3 questions, one per line."
            - Call portkey_client.chat_completion()
            - Parse response into list of 3 strings
            - Also include the original rewritten_query → total 4 queries
            - Set state["multi_queries"] = [q1, q2, q3, q4]
            - Append "🔀 Generated {N} query variants" to state["steps"]
            - Return updated state dict

NOTE: All 4 queries will be used by retriever_node.
      Results are unioned and deduplicated before reranking.

EXAMPLE:
    Input:  "What classification accuracy did the proposed CNN model achieve?"
    Output: [
        "What is the test accuracy of the deep learning model?",
        "How does the proposed architecture perform on the benchmark?",
        "What evaluation metrics does the paper report for the neural network?"
    ]
"""

