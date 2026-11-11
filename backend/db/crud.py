"""
db/crud.py — Database CRUD Operations (Create, Read, Update, Delete)
=====================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
All functions are async and accept an AsyncSession as first argument.

--- Paper CRUD ---
Step 1:  create_paper(db, filename, file_path, ...) → Paper
            - Create a new Paper record with status="uploading"
            - Return the created object

Step 2:  get_paper(db, paper_id) → Paper | None
            - Fetch a paper by its UUID
            - Return None if not found

Step 3:  get_all_papers(db) → list[Paper]
            - Return all papers ordered by created_at DESC

Step 4:  update_paper_status(db, paper_id, status, page_count=None) → Paper
            - Update status field (uploading → processing → ready | failed)
            - Optionally update page_count after extraction

Step 5:  delete_paper(db, paper_id) → bool
            - Delete paper record
            - Also physically delete the PDF file from disk
            - Return True if deleted, False if not found

--- ChatSession CRUD ---
Step 6:  create_session(db, paper_id, title) → ChatSession
Step 7:  get_sessions_for_paper(db, paper_id) → list[ChatSession]
Step 8:  get_session(db, session_id) → ChatSession | None

--- Message CRUD ---
Step 9:  add_message(db, session_id, role, content, retrieved_chunks, model_used, latency_ms) → Message
            - Insert a new Message row (ragas fields all NULL initially)
            - Return created Message object

Step 10: get_messages_for_session(db, session_id) → list[Message]
            - Return messages ordered by created_at ASC (chronological)

Step 11: get_message(db, message_id) → Message | None
            - Fetch a single message by ID
            - Used by evaluation route to return scores

--- RAGAS CRUD ---
Step 12: save_ragas_scores(db, message_id, scores: dict) → Message
            - Find the assistant Message by message_id
            - Update these fields from scores dict:
                * ragas_faithfulness      = scores["faithfulness"]
                * ragas_answer_relevancy  = scores["answer_relevancy"]
                * ragas_context_precision = scores["context_precision"]
                * ragas_overall_score     = scores["overall_score"]
                * ragas_label             = scores["label"]
                * ragas_evaluated_at      = datetime.utcnow()
            - Commit and return updated Message
            - Called from background task after RAGAS evaluation completes

Step 13: get_session_ragas_summary(db, session_id) → dict
            - Fetch all assistant messages for the session that have ragas scores
            - Compute averages: avg_faithfulness, avg_relevancy, avg_precision, avg_overall
            - Return summary dict with score_history list

Step 14: get_paper_ragas_report(db, paper_id) → dict
            - Join Paper → ChatSession → Message
            - Aggregate all evaluated messages across all sessions
            - Return full report: total questions, avg scores, model usage counts, avg latency
"""

