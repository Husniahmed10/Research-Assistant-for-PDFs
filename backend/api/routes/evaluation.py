"""
api/routes/evaluation.py — RAGAS Evaluation Routes
====================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Create APIRouter with prefix="/api/v1/evaluation", tags=["Evaluation"]

--- Route 1: GET /messages/{message_id} ---
Step 2:  GET /messages/{message_id} — Get RAGAS scores for a specific message:
            - Fetch message from DB
            - Return its ragas_scores JSON field:
                {
                  "faithfulness":      0.91,
                  "answer_relevancy":  0.87,
                  "context_precision": 0.79,
                  "context_recall":    null,
                  "overall_score":     0.86,
                  "label":             "Excellent",
                  "evaluated_at":      "2024-11-11T10:30:00"
                }
            - Return 404 if scores not yet computed (evaluation still running)
            - Return 202 if still pending (background task not done yet)

--- Route 2: GET /sessions/{session_id}/summary ---
Step 3:  GET /sessions/{session_id}/summary — Aggregate scores for a whole session:
            - Fetch all messages in the session that have ragas_scores
            - Compute average scores across all evaluated messages
            - Return:
                {
                  "total_messages":    5,
                  "evaluated":         5,
                  "avg_faithfulness":  0.88,
                  "avg_relevancy":     0.84,
                  "avg_precision":     0.76,
                  "avg_overall":       0.83,
                  "label":             "Good",
                  "score_history": [   ← per-message scores for a trend chart
                    {"message_id": "...", "overall_score": 0.91, "created_at": "..."},
                    ...
                  ]
                }

--- Route 3: GET /papers/{paper_id}/report ---
Step 4:  GET /papers/{paper_id}/report — Full evaluation report for a paper:
            - Aggregate RAGAS scores across ALL sessions for this paper
            - Also return: total questions asked, avg latency, model usage breakdown
            - This is the "Report Card" for the paper's Q&A quality
            - Great for demo — shows the RAG system is measurably good
            - Return:
                {
                  "paper_id":           "...",
                  "total_questions":    12,
                  "avg_faithfulness":   0.89,
                  "avg_relevancy":      0.85,
                  "avg_precision":      0.78,
                  "avg_overall":        0.84,
                  "quality_label":      "Good",
                  "openai_calls":       10,
                  "groq_calls":         2,     ← shows fallback happened
                  "avg_latency_ms":     1240
                }

NOTE: This report endpoint is perfect to open during the demo interview.
      It shows measurable proof that your RAG pipeline works well, not just anecdotally.
"""

