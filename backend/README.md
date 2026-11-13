# Backend implementation notes

For complete setup, configuration, API endpoint tables, curl examples and the
architecture diagram, see the [main project README](../README.md). Run backend
commands from this directory with the Python environment active.


## Hybrid retrieval

New uploads fit BM25 on that paper's chunk texts and upsert dense embeddings
alongside sparse values. The fitted parameters are saved atomically as
`uploads/bm25/<paper-id-hash>/bm25_params.json` and loaded on startup. Each paper
keeps separate parameters so later uploads do not change its retrieval behavior.
Standalone calls to `fit_bm25(corpus)` save `uploads/bm25_params.json`.

Chat uses hybrid retrieval. `HYBRID_ALPHA` controls query weighting: `1` uses
dense retrieval, `0` uses sparse retrieval, and the default `0.7` favors dense
similarity. Weighting is applied to query vectors before calling Pinecone.
Old papers without fitted parameters continue using dense search; delete and
re-upload a paper to generate its sparse vectors. Keep the existing dotproduct
index; no new index or embedding model is required. Deletion removes the paper's
saved encoder parameters after its vectors are deleted.

BM25 uses NLTK tokenization resources. On a fresh environment, provision them once:

```powershell
python -m nltk.downloader punkt punkt_tab stopwords
```

Offline hybrid tests (real BM25 encoders, fake Pinecone index):

```powershell
python -m utils.test_hybrid
```

## Reranking and rank fusion helpers

`services.reranker.rerank_chunks(query, chunks, top_n)` tries Cohere's configured
model (default `rerank-english-v3.0`), then FlashRank if credentials are absent or
the API fails. It returns copies of the chunks sorted by `relevance_score` while
preserving their source metadata. Use a thread pool when calling it from an async
route. The chat agent uses these helpers after hybrid retrieval and RRF fusion.

FlashRank's first use needs a model download. Prepare the ignored local cache
before running offline or deploying without network access:

```powershell
python -c "from services.reranker import prepare_flashrank; prepare_flashrank()"
```

Keep `FLASHRANK_CACHE_DIR` (default `.cache/flashrank`) available on the machine
that runs inference. A fresh checkout does not contain these cached weights.

`utils.text_utils.apply_rrf(ranked_lists, k=60)` merges ranked string IDs or chunk
dictionaries. It sums `1 / (k + rank)` using ranks starting at 1, deduplicates IDs,
and adds `rrf_score` to dictionary results without replacing retrieval scores.

```powershell
python -m utils.test_reranking
```

## Agent state and nodes

`agents/state.py` defines the optional state fields and an additive `steps`
reducer. Nodes return partial updates and only their new progress messages.
The ten node modules implement input validation, classification, rewriting,
query expansion, HyDE, hybrid retrieval/RRF, reranking, synthesis, output
validation, and source formatting.

Pass the application index through LangGraph runtime configuration, not state:

```python
config = {"configurable": {"pinecone_index": app.state.pinecone_index}}
```

The NeMo Colang configuration in `guardrails/` runs input/output checks through
Portkey custom actions. Dialogue generation and intent search are disabled for
this validation-only runtime. Brief greetings, thanks and standalone introductions
receive local replies; introductions attached to paper questions use normal
retrieval. Contextual follow-ups and author professional contact details printed
in the paper are allowed. Missing evidence is handled by the answering stage,
not treated as an unsafe question.
Rejected requests receive a refusal; failed or ambiguous model checks report
validation unavailable rather than falsely classifying the question as off-topic.
The named input flows are `check jailbreak`, `check prompt injection`, and
`check off topic`; the output flow is `check harmful content` (also checks
source support). High-confidence instruction overrides and secret-extraction
commands are rejected locally before creating the NeMo runtime or calling any
model/retrieval service. Other requests use the semantic gateway check. Colang
contains example user patterns; explicit Python actions enforce these rails
without invoking model-based intent matching. The configured main model is
`gpt-4o`; validation actions continue to use the saved Portkey configuration.
Paper questions that discuss attacks academically still receive semantic checks.
HyDE passages are retrieval aids and never become source
evidence. Citations use the actual excerpts supplied to synthesis.

`agents/graph.py` exports the compiled `agent`. The chat API invokes it with the
question, paper/session IDs, and the last four stored messages. The graph stops
on rejected input or fatal errors and retries unsafe output only while its
generation budget remains. The default is two generation attempts (one retry),
with a maximum supported budget of three. `retry_count` records completed retries.
Exact name-recall follow-ups use a deterministic lookup of user introductions
from the full SQL session, separately from the four-message LLM context. They
return from the input node without retrieval, PDF citations or RAGAS evaluation.
Names do not carry between sessions, and missing names are not guessed.
Only verified paper answers reach citation formatting. Blocked input returns a refusal;
provider failures and exhausted answer validation do not save partial exchanges.

Chat responses include `answer`, `citations`, `model_used`, `latency_ms`, and
`steps`. Cited sources are grouped by PDF page and retain all associated source
labels and chunk IDs. History reconstructs the same citations from saved excerpts.
Progress steps are returned for the current request; they are not stored in the
message table. `services.simple_rag` remains available for explicit fallback use.

Tracing is configured during application startup, before serving requests.
`setup_langsmith()` runs before importing the chat graph and exports the configured
key, project and endpoint. Set `LANGCHAIN_TRACING_V2=true` and a valid
`LANGCHAIN_API_KEY` to trace graph runs in LangSmith; false or an empty key disables
remote LangSmith tracing. The SDK traces graph/node execution automatically.

`setup_logfire(app)` runs during app construction because instrumentation installs
middleware. It instruments FastAPI, SQLAlchemy's async engine via its sync proxy,
and HTTPX once. Set `LOGFIRE_TOKEN` to export to Logfire; an empty token selects
local-only tracing. Manual service spans record PDF page/chunk counts, estimated
and provider-reported embedding tokens, dense/hybrid search result counts and
latency, and synthesis model/latency. These manual attributes omit paper text,
prompts, embeddings and credentials. Automatic framework traces and LangSmith
graph states can include request/document data.

Install the updated `requirements.txt` to include the instrumentation extras and
compatible OpenTelemetry versions. Run `python -m utils.test_tracing` to verify
instrumentation with an in-memory exporter, without cloud requests.

Verified chat answers with source excerpts are evaluated after the response by a
FastAPI background task. `services/evaluator.py` uses RAGAS with `gpt-4o-mini`
and `text-embedding-3-small`, using `OPENAI_API_KEY` directly. Chat returns the
assistant `message_id`; blocked/no-source replies are marked `skipped`.
Scores live in the additive `message_evaluations` table, created by `init_db()`
on restart without changing existing message columns. Existing messages without
an evaluation remain `not_scheduled`.

Read results with these endpoints (GET requests do not run evaluation):

- `GET /api/v1/evaluation/messages/{message_id}`: saved scores/status, or HTTP 202
  while pending; missing/legacy evaluations return 404, user messages return 400.
- `GET /api/v1/evaluation/sessions/{session_id}/summary`: averages over completed
  answers, status counts and score history. `total_messages` includes both roles.
- `GET /api/v1/evaluation/papers/{paper_id}/report`: all sessions for that paper,
  question count, average answer latency and successful-answer model counts.
  `model_usage` counts saved answer models, not attempted provider/fallback calls.

The overall score weights faithfulness 40%, answer relevancy 35% and precision
25%. Labels are Excellent (>=0.8), Good (>=0.6), Fair (>=0.4), Poor (<0.4).
Missing, failed and non-finite metrics remain null and receive no overall label.
Without a reference answer, RAGAS uses `context_utilization` as the reference-free
precision variant; the API identifies it with `precision_metric`. With a supplied
`ground_truth`, the service uses `context_precision` and includes context recall.
Reports include `precision_metrics` counts to distinguish these variants.

Evaluation/import/provider errors are contained and do not change the chat
response. Background tasks run in-process: a server restart can leave a pending
evaluation unfinished. This is optional quality measurement, not a durable job
queue or a guarantee of correctness. Run `python -m utils.test_evaluation` for
provider-mocked scoring, persistence, failure and report tests.

Both POST chat and SSE use `run_ragas_background()` after committing the answer.
The evaluator runs in a worker thread, writes through `crud.save_ragas_scores()`
using a fresh database session, and cannot turn evaluation failure into chat
failure. Evaluation contexts are the exact reranked excerpts supplied to synthesis.
Poll `/api/v1/evaluation/messages/{message_id}`; HTTP 202 means pending, and
the returned `status` distinguishes completed, failed and skipped evaluations.

Use `GET /api/v1/chat/sessions/{session_id}/stream?query=...` for SSE chat.
It emits `data: <JSON>` frames separated by blank lines:

- `step`: progress as graph nodes finish, including retrieval and reranking.
- `token`: whitespace-preserving pieces of the verified final answer.
- `done`: citations, actual `model_used`, `message_id` and `latency_ms`.
- `error`: a sanitized message and `status_code`; no `done` follows an error.

Progress streams during execution. Answer text is buffered through output safety
checks and citation formatting, then saved before token events begin. Rejected
drafts are never streamed; these token events are final-answer pieces, not raw
provider token events. Both routes persist the same message/history format.
The stream sends keep-alive comments during long operations and disables proxy
buffering/caching. Disconnecting during generation cancels outstanding graph work;
an answer already committed remains saved. SSE persistence uses its own session,
because the request dependency can close before streaming in the pinned FastAPI.

Browser example (close on completion/error to prevent automatic reconnection from
submitting another question):

```javascript
const stream = new EventSource(
  `http://localhost:8000/api/v1/chat/sessions/${sessionId}/stream?query=${encodeURIComponent(query)}`
);
stream.onmessage = ({ data }) => {
  const event = JSON.parse(data);
  if (event.type === "step") showProgress(event.content);
  if (event.type === "token") appendAnswer(event.content);
  if (event.type === "done") {
    stream.close();
    showCitations(event.citations);
    pollEvaluation(event.message_id);
  }
  if (event.type === "error") {
    stream.close();
    showError(event.content);
  }
};
stream.onerror = () => stream.close();
```

Test from PowerShell with an existing ready-paper session:

```powershell
curl.exe -N --get --data-urlencode "query=What are the key results?" "http://localhost:8000/api/v1/chat/sessions/SESSION_ID/stream"
```

Run `python -m utils.test_chat_stream` for SSE, validation, persistence, retry,
disconnect and background-evaluation checks without paid provider requests.

`GET /api/v1/health/` (also available without the trailing slash) checks the SQL
database with `SELECT 1`, reads Pinecone index statistics, and sends a small test
completion through the configured Portkey gateway. Checks run concurrently with
a 10-second timeout per component. It returns `status`, `db`, `pinecone`, `llm`
and a UTC `timestamp`: HTTP 200 for `healthy`, HTTP 503 for `degraded`, with each
component marked `ok` or `error`. Dependency errors are logged without exposing
provider messages or credentials. A successful gateway fallback also counts as
LLM availability; this does not prove the primary provider is available.
The health endpoint makes a real gateway request each time. Use `GET /` for a
lightweight liveness check that does not contact external providers.

All API routers use `ErrorHandlingRoute` to wrap route/dependency/serialization
errors consistently. Unexpected errors return `500 {"detail":"Internal server
error"}` and are logged to Logfire with their type, method and route template.
Global handlers also cover errors outside these routes. Explicit HTTP statuses
are preserved: invalid PDF uploads 400, missing resources 404, request validation
422, processing conflicts 409, invalid generated answers 502 and unavailable
services 503. Validation responses include field locations/messages without
echoing submitted input. SSE errors after headers are sent remain `error` events.

`main.py` configures LangSmith before graph imports, installs Logfire before ASGI
startup, then initializes DB tables, Pinecone, saved BM25 encoders and the uploads
directory in its lifespan. Papers, chat, evaluation and health routers are
registered once during app construction. Failed initialization prevents startup
and disposes the database engine; shutdown clears the stored Pinecone reference.
Run `python -m utils.test_health_errors` to check health probes, error handling,
startup order and failure cleanup with remote services mocked.

```powershell
python -m utils.test_agent_nodes
python -m utils.test_agent_graph
python -m utils.test_chat
```

Tests execute all ten nodes, the real NeMo/Colang runtime, and a small LangGraph
configuration/reducer check. Provider calls are mocked and test tracing disabled.

To check the reported introduction/authorship and paper-assessment examples plus
two rejection cases against the live gateway (four small provider calls):

```powershell
python -m utils.check_input_guard --live
python -m utils.test_guardrails
```
