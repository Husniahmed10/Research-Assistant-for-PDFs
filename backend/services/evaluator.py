"""
services/evaluator.py — RAGAS Evaluation Service
==================================================

WHAT IS RAGAS?
--------------
RAGAS (Retrieval Augmented Generation Assessment) is a framework that
evaluates the quality of your RAG pipeline WITHOUT needing human-labeled
ground truth answers. It scores 4 key dimensions automatically.

THE 4 METRICS WE USE (no ground truth needed):
-----------------------------------------------

1. FAITHFULNESS (0.0 → 1.0)
   "Is the answer factually grounded in the retrieved context?"
   - Checks if every claim in the answer can be found in the context chunks
   - Score 1.0 = every statement is supported by context
   - Score 0.0 = answer is hallucinated / not in context
   - HOW: LLM breaks answer into claims → checks each claim against context

2. ANSWER RELEVANCY (0.0 → 1.0)
   "Does the answer actually address the question asked?"
   - Generates N questions from the answer → checks similarity to original question
   - Score 1.0 = answer directly addresses the question
   - Score 0.0 = answer is off-topic or too vague
   - HOW: Reverse-engineer questions from answer → cosine similarity to original query

3. CONTEXT PRECISION (0.0 → 1.0)
   "Are the retrieved chunks actually relevant to the question?"
   - Checks what fraction of retrieved chunks were actually useful
   - Score 1.0 = all retrieved chunks were relevant
   - Score 0.0 = retrieval pulled irrelevant chunks
   - HOW: LLM judges each chunk's relevance to the question

4. CONTEXT RECALL (0.0 → 1.0)
   "Did we retrieve ALL the information needed to answer?"
   - Checks if the answer contains info that wasn't in the context
   - Score 1.0 = all answer content is traceable to context
   - Score 0.0 = answer needed info that wasn't retrieved
   - NOTE: Requires ground truth — use only if you have reference answers

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------

Step 1:  Import from ragas:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from ragas.metrics import context_recall  # optional — needs ground truth
            from datasets import Dataset

Step 2:  Import OpenAI LLM and embeddings wrappers for RAGAS:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            RAGAS uses these internally to run its evaluation prompts

Step 3:  Write evaluate_rag_response(
                query: str,
                answer: str,
                contexts: list[str],      # text of each retrieved chunk
                ground_truth: str = None  # optional reference answer
            ) -> dict:

            --- Build the RAGAS dataset ---
            data = {
                "question":  [query],
                "answer":    [answer],
                "contexts":  [contexts],   # list of lists (one list per sample)
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            --- Choose metrics based on available data ---
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics.append(context_recall)

            --- Run evaluation ---
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ChatOpenAI(model="gpt-4o-mini"),  # use mini to save costs
                embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
            )

            --- Extract scores ---
            scores = result.to_pandas().iloc[0].to_dict()
            return {
                "faithfulness":      round(scores.get("faithfulness", 0), 3),
                "answer_relevancy":  round(scores.get("answer_relevancy", 0), 3),
                "context_precision": round(scores.get("context_precision", 0), 3),
                "context_recall":    round(scores.get("context_recall", 0), 3) if ground_truth else None,
                "overall_score":     compute_overall(scores)
            }

Step 4:  Write compute_overall(scores: dict) -> float:
            - Weighted average: faithfulness (40%) + answer_relevancy (35%) + context_precision (25%)
            - Return a single 0.0–1.0 score for the UI badge

Step 5:  Write score_to_label(score: float) -> str:
            - 0.8–1.0 → "Excellent"
            - 0.6–0.8 → "Good"
            - 0.4–0.6 → "Fair"
            - 0.0–0.4 → "Poor"
            - Used to display a colored badge in the frontend

Step 6:  Wrap the whole evaluate_rag_response() in try/except:
            - RAGAS calls the LLM multiple times — if it fails, return None scores gracefully
            - Log evaluation time and scores with Logfire

COST ESTIMATE:
    ~3-5 LLM calls per evaluation (using gpt-4o-mini ≈ $0.001 per Q&A evaluation)
    Run as a background task so it doesn't slow down the chat response

WHERE THIS FITS IN THE FLOW:
    User asks question
         ↓
    Agent generates answer  ← main response returned to user immediately
         ↓ (background task)
    RAGAS evaluates answer  ← runs async, stores scores in DB
         ↓
    User can view scores in UI (optional panel)
"""

