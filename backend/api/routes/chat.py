"""
api/routes/chat.py — Chat / Q&A Routes
========================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Create an APIRouter with prefix="/api/v1/chat", tags=["Chat"]

--- Route 1: POST /sessions ---
Step 2:  POST /sessions — Create a new chat session:
            - Body: { paper_id: str }
            - Validate paper exists and status is "ready"
            - Create ChatSession in DB
            - Return: { session_id, paper_id, created_at }

--- Route 2: POST /sessions/{session_id}/message ---
Step 3:  POST /sessions/{session_id}/message — Ask a question:
            - Body: { query: str }
            - Validate session exists
            - Fetch chat history (last 4 messages) from DB
            - Build initial AgentState with query, paper_id, session_id, chat_history
            - Run LangGraph agent: result = await agent.ainvoke(initial_state)
            - Save user message and assistant response to DB (crud.add_message)
            - After saving, trigger RAGAS evaluation as a BackgroundTask:
                background_tasks.add_task(
                    run_ragas_evaluation,
                    message_id   = assistant_message.id,
                    query        = query,
                    answer       = result["final_answer"],
                    contexts     = [chunk["text"] for chunk in result["reranked_chunks"]],
                    db           = db
                )
              run_ragas_evaluation():
                  scores = await evaluator.evaluate_rag_response(query, answer, contexts)
                  await crud.save_ragas_scores(db, message_id, scores)
            - Return immediately (don't wait for RAGAS — it's a background task)
            - Return:
                {
                  answer: str,
                  citations: list[dict],
                  model_used: str,
                  steps: list[str],       ← agent thinking steps for UI display
                  is_safe: bool,
                  latency_ms: int
                }

--- Route 3: GET /sessions/{session_id}/messages ---
Step 4:  GET /sessions/{session_id}/messages — Get full chat history:
            - Fetch all messages for session from DB
            - Return ordered list of { role, content, created_at, citations }

--- Route 4: GET /sessions/{session_id}/stream (Bonus - Streaming) ---
Step 5:  GET /sessions/{session_id}/stream — SSE streaming endpoint:
            - Accept query as query parameter
            - Use Server-Sent Events (SSE) to stream:
                * Agent step updates: "data: {step: '🔍 Searching...', type: 'step'}"
                * Token-by-token answer: "data: {token: 'The', type: 'token'}"
                * Final citations: "data: {citations: [...], type: 'done'}"
            - Use StreamingResponse with media_type="text/event-stream"
"""

