"""
services/pdf_processor.py — PDF Parsing, Text Extraction & Chunking
====================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------

--- Part A: Text Extraction ---
Step 1:  Import pymupdf (fitz) as the primary PDF parser
Step 2:  Write extract_text_from_pdf(file_path: str) → dict:
            - Open PDF with fitz.open(file_path)
            - For each page, extract text + page number
            - Return: { "full_text": str, "pages": [{"page_num": int, "text": str}], "page_count": int }
            - Log extraction time with Logfire span

--- Part B: Semantic Chunking ---
Step 3:  Write chunk_text_fixed(pages: list, chunk_size: int, overlap: int) → list[dict]:
            - Basic approach: sliding window over tokens (use tiktoken to count)
            - Each chunk dict: { "text": str, "page_num": int, "chunk_index": int }

Step 4:  Write chunk_text_semantic(pages: list) → list[dict]:
            - Advanced approach: split at paragraph/section boundaries
            - Detect section headers (regex for "Abstract", "Introduction", etc.)
            - Each chunk corresponds to a logical section
            - Preferred over fixed chunking for research papers

Step 5:  Write apply_parent_child_chunking(chunks: list) → list[dict]:
            - For each large "parent" chunk (512 tokens), create smaller "child" chunks (128 tokens)
            - Each child stores reference to parent chunk id
            - Pinecone indexes child chunks; LLM receives parent context

Step 6:  Write a main process_pdf(file_path: str) → dict entry point:
            - Calls extract_text_from_pdf()
            - Calls chunk_text_semantic() (falls back to fixed if needed)
            - Calls apply_parent_child_chunking()
            - Returns all chunks ready for embedding
"""

