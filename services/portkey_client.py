"""
services/portkey_client.py — Portkey LLM Gateway (Fallback: OpenAI → Groq)
============================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import Portkey client from portkey_ai
Step 2:  Build the fallback config dict:
            {
              "strategy": {"mode": "fallback"},
              "targets": [
                {
                  "virtual_key": PORTKEY_VIRTUAL_KEY_OPENAI,
                  "override_params": {"model": "gpt-4o", "temperature": 0.1}
                },
                {
                  "virtual_key": PORTKEY_VIRTUAL_KEY_GROQ,
                  "override_params": {"model": "llama-3.1-70b-versatile", "temperature": 0.1}
                }
              ]
            }
Step 3:  Initialize Portkey client with API key and config

Step 4:  Write chat_completion(messages: list[dict], stream: bool = False) → str | AsyncGenerator:
            - Call portkey.chat.completions.create(messages=messages, stream=stream)
            - If stream=True, return an async generator that yields tokens (for streaming UI)
            - If stream=False, return the full response text
            - Portkey automatically retries on failure and falls back to Groq
            - Log which provider was actually used (from response headers)

Step 5:  Write a helper get_model_used(response) → str:
            - Portkey adds custom headers showing which provider responded
            - Parse and return "openai" or "groq" for the UI badge display

NOTE: All LLM calls in the project go through THIS client only.
      Never call openai or groq SDKs directly from agent nodes.
"""

