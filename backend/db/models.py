"""
db/models.py — SQLAlchemy Database Models (Table Definitions)
==============================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import Column, String, Integer, DateTime, Text, Float, ForeignKey, JSON from SQLAlchemy
Step 2:  Import Base from db.database
Step 3:  Import uuid, datetime for defaults

--- TABLE 1: Paper ---
Step 4:  Define Paper model (represents an uploaded PDF):
            - id: String (UUID, primary key)
            - filename: String (original filename)
            - file_path: String (local path where PDF is stored)
            - file_size_kb: Float
            - page_count: Integer
            - status: String  → "uploading" | "processing" | "ready" | "failed"
            - full_text: Text (optional — raw extracted text)
            - created_at: DateTime (default = now)
            - updated_at: DateTime (auto-update on change)

--- TABLE 2: ChatSession ---
Step 5:  Define ChatSession model (a conversation about a paper):
            - id: String (UUID, primary key)
            - paper_id: String (ForeignKey → Paper.id)
            - title: String (auto-generated from first question)
            - created_at: DateTime

--- TABLE 3: Message ---
Step 6:  Define Message model (individual messages in a session):
            - id: String (UUID, primary key)
            - session_id: String (ForeignKey → ChatSession.id)
            - role: String → "user" | "assistant"
            - content: Text (the message text)
            - retrieved_chunks: JSON (store which chunks were used — for citation display)
            - model_used: String (openai | groq — which LLM answered)
            - latency_ms: Integer (how long it took)
            - created_at: DateTime

Step 7:  Set up relationships:
            - Paper → ChatSession (one-to-many)
            - ChatSession → Message (one-to-many)
"""

