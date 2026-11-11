"""
services/summarizer.py — Structured Paper Summary Generator
=============================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import portkey_client for LLM calls
Step 2:  Define the SUMMARY_PROMPT template with clear structure instructions:
            - System prompt: "You are a research paper analyst..."
            - User prompt: includes full paper text (or first N tokens) + format instructions

Step 3:  Write generate_summary(full_text: str) → dict:
            - Truncate full_text to fit model context (use tiktoken, max ~12000 tokens)
            - Build the prompt requesting structured output with these exact sections:
                * title_and_authors: str
                * abstract: str
                * problem_statement: str
                * methodology: str
                * key_results: str
                * conclusion: str
            - Call portkey_client.chat_completion(messages)
            - Parse the LLM response (expect JSON or structured text)
            - Return as a dict with the 6 sections
            - Wrap in Logfire span with paper length and token count

Step 4:  Write parse_summary_response(raw_response: str) → dict:
            - Try JSON parsing first (if LLM returns JSON)
            - Fallback: regex-based section extraction
            - Ensure all 6 required sections are present
            - Return dict ready for API response

NOTE: Summary is generated ONCE on upload and stored in the DB (or returned directly).
      It does NOT use Pinecone — it uses the full text directly.
"""

