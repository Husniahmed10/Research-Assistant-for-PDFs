"""
utils/text_utils.py — Text Processing Helper Utilities
=======================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Write count_tokens(text: str, model: str = "gpt-4o") → int:
            - Use tiktoken to count tokens for a given model
            - Used to ensure chunks and prompts don't exceed context limits

Step 2:  Write truncate_to_tokens(text: str, max_tokens: int, model: str) → str:
            - Truncate text to fit within max_tokens
            - Preserves full sentences (don't cut mid-sentence)
            - Used in summarizer.py to fit full paper text into context

Step 3:  Write clean_text(text: str) → str:
            - Remove excessive whitespace, null bytes, control characters
            - Normalize unicode characters
            - Used to clean raw PDF-extracted text before chunking

Step 4:  Write extract_section_headers(text: str) → list[str]:
            - Regex to detect common research paper section headers:
                Abstract, Introduction, Related Work, Methodology,
                Experiments, Results, Discussion, Conclusion, References
            - Returns list of detected section names
            - Used by semantic chunker in pdf_processor.py

Step 5:  Write apply_rrf(ranked_lists: list[list], k: int = 60) → list:
            - Reciprocal Rank Fusion algorithm
            - Input: multiple ranked lists of chunk IDs
            - Output: single merged ranked list
            - Formula: score(doc) = sum(1 / (rank + k)) for each list
            - Used by retriever node to merge multi-query results
"""

