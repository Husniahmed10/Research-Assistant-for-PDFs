"""
agents/nodes/synthesizer.py — Answer Generation Node (Node 8)
==============================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import portkey_client and AgentState

Step 2:  Define the QA_SYSTEM_PROMPT:
            "You are a research assistant specialized in analyzing academic papers.
             Answer the user's question using ONLY the provided context excerpts.
             If the answer is not in the context, say so explicitly.
             Always cite the page number when referencing specific information.
             Be concise, accurate, and scholarly in tone."

Step 3:  Write synthesizer_node(state: AgentState) → dict:
            - Extract state["reranked_chunks"], state["query"], state["chat_history"]

            - Build context string from reranked chunks:
                context = "\n\n".join([
                    f"[Page {chunk['page_num']}]: {chunk['text']}"
                    for chunk in state["reranked_chunks"]
                ])

            - Build messages list:
                * System message: QA_SYSTEM_PROMPT
                * Include last 4 messages from chat_history (for multi-turn memory)
                * User message: f"Context:\n{context}\n\nQuestion: {state['query']}"

            - Call portkey_client.chat_completion(messages, stream=False)
            - Track which provider was used (from response metadata)
            - Set state["final_answer"] = response text
            - Set state["model_used"] = "openai" or "groq"
            - Append "💬 Answer generated using {model_used}" to state["steps"]
            - Return updated state dict
"""

