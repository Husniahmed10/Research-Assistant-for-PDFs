"""
agents/nodes/input_guard.py — NeMo Guardrails Input Safety Check (Node 1)
==========================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import nemoguardrails and AgentState
Step 2:  Initialize NeMo LLMRails with config from guardrails/ folder:
            rails_config = RailsConfig.from_path("guardrails/")
            rails = LLMRails(rails_config)

Step 3:  Write input_guard_node(state: AgentState) → dict:
            - Extract state["query"]
            - Run the query through NeMo input rails:
                result = await rails.generate_async(
                    messages=[{"role": "user", "content": state["query"]}]
                )
            - If NeMo rails blocked the message (returned a rejection):
                * Set state["is_safe"] = False
                * Set state["rejection_reason"] = "Jailbreak attempt detected" / "Off-topic query"
                * Set state["final_answer"] = NeMo's polite rejection message
            - If safe:
                * Set state["is_safe"] = True
            - Append "✅ Input safety check passed" or "🚫 Input blocked" to state["steps"]
            - Return updated state dict

NOTE: NeMo uses the Colang rules defined in guardrails/main.co to decide.
      The rails config is at guardrails/config.yml.
      This node MUST run first before any LLM call.
"""

