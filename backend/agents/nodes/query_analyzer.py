"""
agents/nodes/query_analyzer.py — Query Type Classification (Node 2)
====================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import portkey_client and AgentState

Step 2:  Write query_analyzer_node(state: AgentState) → dict:
            - Extract state["query"]
            - Build a classification prompt:
                "Classify this question into one of: factual, summary, comparative, definition, methodology.
                 Question: {query}
                 Return only the category name."
            - Call portkey_client.chat_completion() with the prompt
            - Parse the response to get query_type
            - Set state["query_type"] = query_type
            - Append "🔍 Query classified as: {query_type}" to state["steps"]
            - Return updated state dict

            Query types guide downstream behavior:
            - "summary"     → might skip retrieval and use full text
            - "factual"     → needs precise retrieval
            - "methodology" → targets methodology section specifically
            - "comparative" → might need multiple chunks from different sections
"""

