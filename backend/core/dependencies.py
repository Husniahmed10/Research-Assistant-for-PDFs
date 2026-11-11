"""
core/dependencies.py — Shared FastAPI Dependencies (Dependency Injection)
==========================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Create an async `get_db()` dependency function:
            - Yields an AsyncSession from SQLAlchemy
            - Automatically closes session after request completes
            - Used as: db: AsyncSession = Depends(get_db) in route functions
Step 2:  Create a `get_pinecone_index()` dependency:
            - Returns the initialized Pinecone index instance
            - Used in retrieval routes
Step 3:  Create a `get_settings()` dependency:
            - Returns the global settings object
            - Useful for injecting config into routes/services
Step 4:  (Optional) Create a `get_current_user()` dependency:
            - Decodes JWT token from Authorization header
            - Returns user info (if adding auth later)
"""

