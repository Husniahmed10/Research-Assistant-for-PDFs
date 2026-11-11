"""
api/routes/health.py — Health Check Route
==========================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Create APIRouter with prefix="/api/v1/health", tags=["Health"]

Step 2:  GET / — Health check endpoint:
            - Check SQLite DB connection
            - Check Pinecone index connectivity
            - Check OpenAI API reachability (via Portkey)
            - Return:
                {
                  status: "healthy" | "degraded",
                  db: "ok" | "error",
                  pinecone: "ok" | "error",
                  llm: "ok" | "error",
                  timestamp: datetime
                }
            - Used by frontend to show service status banner
            - Also useful for Docker/K8s readiness probes
"""

