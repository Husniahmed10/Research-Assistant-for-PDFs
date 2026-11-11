"""
agents/nodes/output_guard.py — NeMo Guardrails Output Safety Check (Node 9)
============================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import nemoguardrails rails instance and AgentState

Step 2:  Write output_guard_node(state: AgentState) → dict:
            - Extract state["final_answer"]

            --- Check 1: Hallucination / Groundedness ---
            - Verify the answer only references information present in state["reranked_chunks"]
            - Simple approach: check if key claims in the answer appear in chunk texts
            - Advanced: use NeMo's output rails with a fact-checking prompt

            --- Check 2: Harmful Content ---
            - Run the answer through NeMo output rails:
                result = await rails.generate_async(
                    messages=[
                        {"role": "user",      "content": state["query"]},
                        {"role": "assistant", "content": state["final_answer"]}
                    ]
                )
            - NeMo will flag if the output contains harmful/toxic content

            --- Decision ---
            - If output passes all checks:
                * Set state["is_output_safe"] = True
            - If output fails:
                * Set state["is_output_safe"] = False
                * The graph will route back to synthesizer for regeneration

            - Append "✅ Output safety verified" or "⚠️ Output flagged, regenerating" to state["steps"]
            - Return updated state dict
"""

