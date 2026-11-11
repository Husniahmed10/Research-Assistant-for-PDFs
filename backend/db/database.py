"""
db/database.py — SQLAlchemy Async Database Setup
==================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import create_async_engine, AsyncSession, declarative_base from SQLAlchemy
Step 2:  Read DATABASE_URL from settings (e.g., sqlite+aiosqlite:///./research_assistant.db)
Step 3:  Create async engine:
            engine = create_async_engine(DATABASE_URL, echo=settings.DEBUG)
Step 4:  Create AsyncSessionLocal factory:
            AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
Step 5:  Create Base = declarative_base() — all models will inherit from this
Step 6:  Write an async `init_db()` function:
            - Calls Base.metadata.create_all(bind=engine)
            - Creates all tables if they don't exist
            - Called at app startup in main.py
Step 7:  Export: engine, AsyncSessionLocal, Base, init_db
"""

