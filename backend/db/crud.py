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
Step 10: get_messages_for_session(db, session_id) → list[Message]
            - Return messages ordered by created_at ASC (chronological)
"""

