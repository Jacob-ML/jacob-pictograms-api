"""FastAPI application — image similarity search endpoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI

from src.config import config
from src.store import EmbeddingStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global state (initialised once at startup)
# ---------------------------------------------------------------------------

store: EmbeddingStore | None = None
_openai_client: OpenAI | None = None


def _init_openai() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.openrouter_api_key,
    )


# ---------------------------------------------------------------------------
# Lifespan — load data once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, _openai_client

    logger.info("Loading dataset …")
    store = EmbeddingStore()
    logger.info(
        "Dataset loaded — %d items in memory.", store.count()
    )

    _openai_client = _init_openai()
    logger.info("OpenRouter client ready.")

    yield

    del store
    store = None
    del _openai_client
    _openai_client = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pictogram Search",
    description=(
        "Upload-free image search powered by semantic embeddings.  "
        "Send a short text query and receive the most relevant pictogram."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/search")
async def search(
    q: str = Query(
        description="Short descriptive query (e.g. 'shopping cart').",
        min_length=1,
        max_length=200,
    ),
):
    """Return the most semantically relevant image for *q*.

    The query text is embedded via OpenRouter, dot-producted against all
    stored embeddings, and the best match is returned as a base64 PNG.
    """
    if store is None or _openai_client is None:
        raise HTTPException(503, "Service not yet initialised.")

    # --- embed query via OpenRouter ----------------------------------------
    resp = _openai_client.embeddings.create(
        input=q,
        model=config.embedding_model,
        dimensions=256,
    )
    query_vec = np.array(resp.data[0].embedding, dtype=np.float32)

    # --- find best match (vectorised dot product) --------------------------
    idx = store.search(query_vec)
    result = store.get_result(idx)

    logger.info("Query %r → matched id=%s", q, result["id"])
    return result


@app.get("/health")
async def health() -> dict:
    """Simple readiness probe."""
    if store is None:
        raise HTTPException(503, "Not ready yet.")
    return {"status": "ok", "items": store.count()}
