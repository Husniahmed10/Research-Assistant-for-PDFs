# PaperLens frontend

A Next.js App Router and TypeScript research workspace for the existing FastAPI
backend. The application lives entirely in `frontend/`; backend source and
credentials are unchanged.

## Run locally

Use Node.js **22.13 or newer** (an active Node LTS release is recommended) and npm.
Next.js itself supports older Node versions, but the installed lint tooling needs
22.13+. The initial local checks also ran successfully on this machine’s 22.12.0,
which prints an npm engine warning; update Node to meet the supported minimum.

Start the backend in one terminal, following the root README for credentials,
database, Pinecone, and model resources:

```powershell
conda activate redoc
cd backend
uvicorn main:app --reload
```

From the repository root, open a second terminal:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

On macOS/Linux, use `cp .env.example .env.local` for the copy step. Preserve an
existing `.env.local` if you have already configured it.

Visit **http://localhost:3000**. The API should run at **http://localhost:8000**.
Use `localhost`, rather than `127.0.0.1`, for the frontend: the existing backend
CORS policy permits `http://localhost:3000`.

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_MAX_FILE_SIZE_MB=50
```

The file-size setting must match `MAX_FILE_SIZE_MB` in `backend/.env` (currently
50 MB). The server remains authoritative. These variables are public browser
configuration, embedded at build time. Never add provider API keys, database
URLs, Logfire tokens, or LangSmith credentials to frontend environment variables.
Restart development or rebuild production after changing configuration.

For a production build:

```powershell
npm run build
npm start
```

For hosting, set the public API URL before building, serve both applications over
HTTPS, and configure the backend’s allowed origins for your real frontend host.
No authentication exists in the current backend; the frontend uses its existing
access model. Deployment access control requires corresponding backend support.

## Using the workspace

- **Library:** upload one PDF at a time; search, filter and sort real papers.
  Upload and background processing are distinct. Processing papers refresh every
  2.5 seconds while the window is active. Failed papers can be deleted and
  re-uploaded; processing papers cannot be deleted.
- **Summary:** read all six structured summary sections and move a suggested
  question into chat. Questions are only submitted when you press Send.
- **Chat:** create a conversation, send follow-ups in that same session, inspect
  actual workflow steps, copy answers, and reopen saved history.
- **Evidence:** click a source label or page chip to inspect its supplied excerpt.
  Source labels map through `source_ids`; multiple labels can refer to one page.
- **Quality:** inspect per-answer evaluations, session summaries, and paper-level
  reports. Pending evaluation polls for up to 90 seconds. Refresh explicitly if
  it takes longer. Unavailable metrics stay unavailable rather than becoming zero.
- **Themes:** light/dark preference persists in this browser. Dialogs support
  Escape, focus trapping and focus restoration; tabs support arrow navigation.

## API integration

| Feature                | Backend API                                                |
| ---------------------- | ---------------------------------------------------------- |
| Connection check       | `GET /` every 60 seconds while active                      |
| Library and processing | `GET /api/v1/papers/`                                      |
| Upload                 | `POST /api/v1/papers/upload`, multipart field `file`       |
| Summary                | `GET /api/v1/papers/{id}/summary`                          |
| Delete                 | `DELETE /api/v1/papers/{id}`                               |
| Create conversation    | `POST /api/v1/chat/sessions`, JSON `{ "paper_id": "..." }` |
| Saved conversation     | `GET /api/v1/chat/sessions/{id}/messages`                  |
| Streamed question      | `GET /api/v1/chat/sessions/{id}/stream?query=...`          |
| Answer evaluation      | `GET /api/v1/evaluation/messages/{id}`                     |
| Conversation quality   | `GET /api/v1/evaluation/sessions/{id}/summary`             |
| Paper report           | `GET /api/v1/evaluation/papers/{id}/report`                |

The deep health endpoint is deliberately not polled because it calls providers.
The frontend does not need an extra proxy, Next.js API routes, or backend changes.

### Streaming and interruption recovery

Streaming uses `fetch` and `ReadableStream`, not automatically reconnecting
EventSource. The parser supports UTF-8 and JSON split across network chunks,
CR/LF framing, multiple data lines, and heartbeat comments. It requires a final
`done` event and handles both HTTP failures and in-band SSE errors.

The backend completes retrieval, generation, safety checks, and persistence before
streaming validated answer text. The workflow activity panel reflects real step
events during the wait; no artificial token generation or processing percentages
are shown.

Switching between Summary, Chat and Answer quality keeps an active chat request
and unsent draft alive. Hidden chat panels do not scroll the visible page or keep
an evidence drawer open. Leaving the paper still disconnects the browser stream.

Stop disconnects the browser stream. It does **not** guarantee cancellation of
backend generation. After an interruption, the UI checks authoritative history.
If the new user/assistant pair is already saved, it displays that pair without
replaying the request. Otherwise it offers Check saved history and an explicit
edit/retry action after a successful history read. No automatic POST fallback or
question retry occurs. Unresolved requests are remembered in memory across
in-app navigation. After a full browser reload, inspect saved history before
manually repeating a recently interrupted question.

### Current backend boundaries

- There is no session-list endpoint. Only session IDs, labels and timestamps are
  indexed in localStorage, namespaced by API URL. Message content is fetched from
  the backend and held in memory. Clearing browser storage loses conversation
  discovery links, although SQL history remains on the backend. Another browser
  cannot automatically discover these conversations.
- There is no PDF-serving endpoint. Evidence uses real retrieved text and page
  references; there is no fabricated PDF URL or filesystem-path exposure.
- Evaluation `202` means pending; `404` can mean not scheduled. Failed/skipped
  results are valid terminal states. Context utilization is labelled as a
  reference-free estimate rather than golden-reference context precision.
- Model, latency and citations come from the API. Per-run workflow steps appear
  during the live run; the history endpoint does not persist step timelines.
- The backend supplies no upload-processing percentage, processing cancellation,
  authentication, conversation renaming/deletion, or multi-paper chat API.

## Source layout

```text
src/app/                  App Router pages, boundaries, global styling
src/components/           Shell, providers, Markdown, accessible UI primitives
src/features/library/     Upload, library, deletion
src/features/paper/       Paper workspace and structured summary
src/features/chat/        Stream lifecycle, messages, evidence panel
src/features/quality/     Evaluation polling and report cards
src/lib/                  Typed API, SSE parsing, query hooks, local session index
e2e/                      Mocked API browser integration tests
```

Tailwind CSS 4 and shared CSS variables provide the visual system. Accessible
dialogs use Radix primitives, following the same primitives used by shadcn/ui.
TanStack Query owns server state. Markdown disables raw HTML, remote images,
and unsafe link protocols. Next Themes owns the color preference; Sonner handles
feedback. No frontend data is sent directly to model providers.

## Checks

```powershell
npm run lint
npm run typecheck
npm test
npm run build
npm run test:e2e
```

Browser tests use installed Microsoft Edge on Windows. On macOS/Linux install
Playwright Chromium with `npx playwright install chromium` first. To use Chromium
instead of Edge on Windows:

```powershell
npx playwright install chromium
$env:PLAYWRIGHT_CHANNEL = "chromium"
npm run test:e2e
```

Tests start an isolated Next.js development server on port 3000 if one is not
already running. Keep that port free or use the same frontend configuration.
External API requests are intercepted by Playwright; no paid provider calls are
made. Browser tests exercise upload, processing, session creation, SSE answers,
grouped citations, pending-to-completed evaluations, reload/history restoration,
deletion, interruption reconciliation, keyboard dialogs, and persisted themes on
desktop and mobile. These are **mocked integration checks**, not live OpenAI,
Pinecone, Portkey, or RAGAS verification. Unit tests cover framing, UTF-8, error
events, no-reconnect behavior, reconciliation, source mapping and validation.

The scoped minimatch override in package.json upgrades the lint parser’s glob
dependency to the patched 11.x release. Keep the lockfile and override together;
the normal lint, TypeScript and build checks validate this combination.

## Troubleshooting

- **API offline:** check `GET http://localhost:8000/`, your public API URL, and CORS.
  Backend startup may fail if Pinecone or its credentials are unavailable.
- **Paper stays processing:** background work is not durable across backend
  restarts. Inspect backend logs; do not mistake an indefinite wait for progress.
- **Chat unavailable:** the paper must be ready; missing sessions return 404.
  Create a new conversation if a saved reference no longer exists.
- **Quality unavailable:** evaluations depend on separate provider credentials
  and can fail without affecting the answer. Conversation-only replies may skip it.
  Expand an answer’s evaluation for a safe failure reason. The backend pins
  PyArrow 24 and Pandas 2.2 for its NumPy 1.26 / RAGAS stack; PyArrow 26 cannot
  import with that NumPy version. After reinstalling backend requirements,
  restart the backend and ask a new paper question. Old failed scores are not
  recomputed by refreshing the report.
- **Stream interrupted:** use Check saved history before manually sending again.
- **Hydration warnings with `bis_skin_checked`, `bis_register`, or injected
  `__processed_*` attributes:** a browser extension modified the HTML before
  React loaded. Disable that extension on localhost or use a clean browser
  profile. See [Next.js hydration troubleshooting](https://nextjs.org/docs/messages/react-hydration-error).
  The app does not suppress these warnings globally or disable server rendering.
- **LangSmith 403 in backend logs:** remote trace ingestion was denied. Check
  your LangSmith key and project/workspace access, or temporarily set
  `LANGCHAIN_TRACING_V2=false` in `backend/.env` and restart the backend while
  correcting access. This setting disables remote tracing, not chat generation.
- **Browser storage blocked:** the UI warns that newly created chat links last
  only for the current visit; the backend still stores their messages.
