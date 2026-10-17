import asyncio
import json
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # must run before llm_client reads LLM_* env vars

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from openai import APIStatusError  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import guardrails  # noqa: E402
import rag  # noqa: E402
import security  # noqa: E402
from llm_client import answer_stream  # noqa: E402

REFRESH_EVERY_S = 60 * 60
logger = logging.getLogger("weather-rag")


async def _refresh_loop():
    """Re-fetches weather hourly so a long-running server doesn't go stale."""
    while True:
        await asyncio.sleep(REFRESH_EVERY_S)
        try:
            await asyncio.to_thread(rag.build_index, True)
        except Exception as err:  # keep serving the old snapshot rather than dying
            print(f"weather refresh failed, keeping previous data: {err}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not security.auth_enabled():
        print("WARNING: APP_API_KEY is not set -- /api/ask is open to anyone who can reach this server.")
    await asyncio.to_thread(rag.build_index)
    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Refuse oversized bodies before they are read. (A client that omits
    Content-Length bypasses this; put a proxy with a body limit in front for that.)"""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > guardrails.MAX_BODY_BYTES:
        return JSONResponse({"error": "request body too large"}, status_code=413)
    return await call_next(request)


class AskRequest(BaseModel):
    question: str = ""


def _line(event: dict) -> str:
    return json.dumps(event) + "\n"


def public_error(err: Exception) -> str:
    """What the client may see. Full details go to the server log only -- provider
    errors can carry organisation ids, quotas and internal paths."""
    logger.exception("ask failed", exc_info=err)
    if isinstance(err, APIStatusError) and err.status_code in (413, 429):
        return "the language model is busy or the question was too large; please try again shortly"
    if isinstance(err, APIStatusError) and err.status_code in (401, 403):
        return "the server could not authenticate with the language model"
    if isinstance(err, RuntimeError) and "LLM_API_KEY" in str(err):
        return "the server is not configured with a language-model key"
    return "something went wrong while answering; please try again"


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    """Streams newline-delimited JSON events: one `retrieved`, then any number of
    `tool` and `token` events, then `done` (or `error`). Plain `def`: the embedding
    and LLM calls block, so FastAPI runs this in a threadpool."""
    blocked = security.guard(request)
    if blocked:
        return blocked
    question, problem = guardrails.screen(req.question)
    if problem:
        # `error` (not FastAPI's default `detail`) is what frontend/src/api.ts reads.
        return JSONResponse({"error": problem}, status_code=400)

    def events():
        try:
            retrieved = rag.retrieve(question)
            yield _line({"type": "retrieved", "retrieved": [rag.to_dict(d) for d in retrieved]})
            for event in answer_stream(question, retrieved):
                yield _line(event)
            yield _line({"type": "done"})
        except Exception as err:  # surface a safe message to the UI, even mid-stream
            yield _line({"type": "error", "error": public_error(err)})

    return StreamingResponse(events(), media_type="application/x-ndjson")
