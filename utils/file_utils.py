"""
utils/file_utils.py — File System Helper Utilities
====================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Write save_upload_file(upload_file: UploadFile, paper_id: str) → str:
            - Generate filename: f"{paper_id}_{upload_file.filename}"
            - Create uploads/ directory if it doesn't exist
            - Write file bytes to uploads/{filename}
            - Return the full file path string

Step 2:  Write delete_upload_file(file_path: str) → bool:
            - Check if file exists
            - Delete it using os.remove()
            - Return True if deleted, False if not found

Step 3:  Write get_file_size_kb(file_path: str) → float:
            - Return file size in kilobytes

Step 4:  Write validate_pdf(file: UploadFile) → bool:
            - Check content_type == "application/pdf"
            - Check file size < MAX_FILE_SIZE_MB
            - Optionally: read first 4 bytes and check for PDF magic bytes (%PDF)
            - Raise HTTPException if invalid
"""

