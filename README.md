# Weather RAG — small retrieval-augmented app (Python backend, React/TS frontend)

Ask weather questions in plain English ("will it rain in Chennai this week?",
"is it a good time to visit Bangkok?", "what was London like a year ago?"). The
backend retrieves the matching live weather notes from a small in-memory index
and lets an LLM (Groq by default) answer from them, calling tools for anything
the notes can't cover. React/TypeScript frontend, Python (FastAPI) backend.

**Live weather.** `backend/data.py` fetches ~100 major cities (Asia, Europe,
North America, South America, Africa, Oceania) from Open-Meteo (free, no key)
at startup, and builds three notes per city: the 7-day forecast, the past 7
days, and the same week one year ago. Rain-day counts are precomputed in code
and written into each note (LLMs miscount). The whole "database" is an
in-memory numpy matrix — this is about seeing the RAG mechanism clearly.

**Retrieval.** A question that names a city (aliases like "Bengaluru" work) gets all
of that city's notes, in order of mention. A question that names no city gets no
notes at all: a handful of "closest" notes out of ~100 cities made the model rank
from a sample, so those questions (regions, rankings) go through the `rank_cities`
tool instead. See `retrieve()` in `backend/rag.py`.

## How it's wired together

```
React/TS UI --POST /api/ask (streamed)--> FastAPI server
                                     |
                         0. screen the question (length, injection phrasing), rate-limit, check the key
                         1. retrieve the notes for any city named in the question
                         2. hand those notes, as context, to the LLM
                         3. the LLM answers from them, OR calls a tool:
                              rank_cities       "rainiest Asian cities this week"
                              get_weather_range "Pune, last Christmas" (any city, any date)
                              travel_advice     "good time to visit Bangkok?" (5-yr climate normals)
                         4. tokens stream back as they are generated
```

This is RAG plus a short tool-calling loop. Retrieval grounds the common
questions in one cheap step; tools handle what notes can't (rankings, arbitrary
dates, untracked cities). Every number is computed in code (`backend/tools.py`,
`backend/climate.py`); the model only explains it.

**Why in-memory numpy and not a vector DB?** ~300 short notes embed in seconds,
and the accuracy wins in this project came from structured lookups and tools,
not nearest-neighbour search. Move to a vector store (pgvector if you already run
Postgres; sqlite-vec or Qdrant otherwise) when you have tens of thousands of
chunks, need metadata-filtered search at scale, or can't afford re-embedding on
startup.

Embeddings are computed locally via Hugging Face `sentence-transformers`
(`all-MiniLM-L6-v2`, ~90MB, downloads once from the HF Hub, cached after) — no Voyage/OpenAI
embeddings API key needed. Embedding and answering are different models
doing different jobs: the local model finds the right notes, the LLM writes
the answer.

The answer step is provider-agnostic (`backend/llm_client.py`): any
OpenAI-compatible API works. It defaults to Groq; set `LLM_API_KEY` in
`backend/.env`, and optionally `LLM_MODEL` / `LLM_BASE_URL` to switch.

## Run it

```bash
cd backend
copy .env.example .env   # fill in your LLM_API_KEY (Groq key)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8080   # first run downloads the embedding model (~25s startup)
```

```bash
cd frontend
npm install
npm run dev             # http://localhost:5173
```

On macOS/Linux use `cp` and `source .venv/bin/activate` instead of `copy` and
`.venv\Scripts\activate`. If you set `APP_API_KEY` (see below), also put
`VITE_API_KEY=<same value>` in `frontend/.env.local`.

## Protecting it

Every question costs LLM quota, so `/api/ask` is gated (`backend/security.py`):

- **API key.** Set `APP_API_KEY` in `backend/.env` and clients must send it as an
  `X-API-Key` header (the frontend does this from `VITE_API_KEY` in `frontend/.env.local`).
  Unset means open, with a warning at startup. Vite bakes `VITE_*` values into the JS
  bundle, so this gates casual abuse on localhost only; for a public deployment put real
  auth (sessions/OAuth) in front.
- **Rate limit.** 10 requests/minute and 200/day per client address by default
  (`RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY`). Over the limit gets `429` with `Retry-After`.
  Failed key attempts count too, so the key can't be brute-forced quickly. It is
  in-memory, i.e. per process: fine for one uvicorn worker, use Redis or a gateway for more.

**Guardrails** (`backend/guardrails.py`, checked before any LLM or embedding work):
questions are capped at 500 characters and request bodies at 8 KB, control and
invisible characters are stripped, and well-known instruction-override and
secret-fishing phrasings are rejected (after Unicode/zero-width normalisation).
Be honest about what that is: a keyword screen can be rephrased around. The real
controls are structural: secrets never enter the model's context, tools are
read-only with validated arguments (enum values, clamped limits, length-capped city
names, dates no earlier than 1940), the system prompt restricts scope and tells the
model to treat the question and all note/tool text as data, and clients only ever
see generic error messages (details go to the server log).

**Staying inside Open-Meteo's free tier** (600 calls/min, 10,000/day), in `backend/data.py`:
outbound calls are spaced at least 0.25 s apart; `429`s and timeouts back off and honour
`Retry-After`; user-triggered lookups (tools, geocoding, climate) share a daily budget
(`UPSTREAM_ON_DEMAND_PER_DAY`, default 500) so chat traffic can't starve the hourly
refresh (~8 calls/hour); results are cached (date ranges, geocoding, climate normals on
disk, the snapshot on disk); and a question can make at most 6 tool calls.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest                 # ~2 s, no network, no LLM, no embedding model
```

They cover the tools (ranking and tie-breaks, date ranges, archive/forecast splitting,
travel advice, argument validation), city detection and retrieval, the climate maths,
the rate limiter and auth end to end, the input guardrails (including that ordinary
weather questions are not blocked), error sanitising, and the Open-Meteo politeness
layer. They do not call the LLM, so they say nothing about answer quality.

## Try it

- "For how many days is it going to rain in Chennai?" — uses the precomputed rain-day count.
- "What was the weather in London a year ago?" — answered from the archive note.
- "Which Asian cities will be rainiest this week?" — calls `rank_cities`.
- "Is it the best time to visit Bangkok now?" — calls `travel_advice` (climate normals vs this week's forecast).
- "How much did it rain in Pune last Christmas?" — Pune isn't tracked; `get_weather_range` geocodes it.
- Tool calls show above each answer; expand "retrieved" to see which notes
  and similarity scores it was grounded in.

## Two real repos worth reading (not included here, just pointers)

- **github.com/vercel/ai-chatbot** — production-grade Next.js chat app with
  streaming responses, multiple model providers. Look at how they handle
  streaming and message persistence; more than this demo needs, good to see
  what "more" looks like.
- **github.com/assistant-ui/assistant-ui** (`@assistant-ui/react` on npm) —
  a React/TypeScript component library specifically for AI chat UIs. Worth
  skimming for how they model message state and streaming in React/TS
  idioms, even if you don't pull in the library itself.

## Known limitations

- Each question is answered on its own: no conversation memory, so "and Delhi?" won't work.
- The model (`gpt-oss-20b` by default) can still misread or swap numbers that the tools
  returned correctly; there is no automated check of answer quality yet.
- Weather refreshes hourly (and is cached in `backend/.cache/`, git-ignored, so restarts are fast); the
  notes themselves only cover the forecast, past week and the week a year ago. Anything else goes through tools.
- City/region detection for retrieval is keyword matching; tools accept any city via Open-Meteo's geocoder.
- The travel comfort score is a crude heuristic (temperature band + rain days). It ignores humidity,
  crowds, prices and events, and the tool output says so.
- Chat history is saved in the browser's localStorage only (last 30 exchanges), not on the server.
- No map UI: it would add a tile/Leaflet dependency without making answers better.
- The "database" is ~300 in-memory vectors (see the note above on when that stops being enough).

## License

MIT, see [LICENSE](LICENSE). Weather data is from [Open-Meteo](https://open-meteo.com)
(free for non-commercial use; check their terms before using it commercially).
