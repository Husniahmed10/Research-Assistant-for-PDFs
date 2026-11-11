"""
api/routes/papers.py — Paper Upload & Management Routes
========================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Create an APIRouter with prefix="/api/v1/papers", tags=["Papers"]

--- Route 1: POST /upload ---
Step 2:  POST /upload — Upload a PDF and process it:
            a) Validate file type (must be application/pdf)
            b) Validate file size (< MAX_FILE_SIZE_MB)
            c) Save file to uploads/ folder with a UUID filename
            d) Create a Paper record in SQL DB with status="uploading"
            e) Start background processing:
                - pdf_processor.process_pdf() → extract text + chunks
                - embedder.embed_chunks() → get dense vectors
                - pinecone_client.upsert_chunks() → index in Pinecone
                - summarizer.generate_summary() → create structured summary
                - Update Paper status to "ready" in DB
            f) Use FastAPI BackgroundTasks for async processing
            g) Return: { paper_id, filename, status: "processing" }

--- Route 2: GET /{paper_id}/status ---
Step 3:  GET /{paper_id}/status — Poll processing status:
            - Fetch paper from DB
            - Return: { paper_id, status, page_count }

--- Route 3: GET /{paper_id}/summary ---
Step 4:  GET /{paper_id}/summary — Get structured summary:
            - Fetch paper from DB
            - Return the 6-section summary dict

--- Route 4: GET / ---
Step 5:  GET / — List all uploaded papers:
            - Fetch all papers from DB ordered by created_at DESC
            - Return list of paper metadata

--- Route 5: DELETE /{paper_id} ---
Step 6:  DELETE /{paper_id} — Delete a paper:
            - Delete PDF file from disk
            - Delete all Pinecone vectors for this paper
            - Delete all DB records (messages, sessions, paper)
            - Return: { message: "deleted" }
"""

