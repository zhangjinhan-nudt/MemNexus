"""Local OpenAI-compatible Qwen3 embedding service for MemNexus experiments."""

import argparse
from functools import lru_cache
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


class EmbeddingRequest(BaseModel):
    model: str = MODEL_NAME
    input: str | list[str] = Field(min_length=1)
    encoding_format: str = "float"
    input_type: Literal["query", "document"] = "document"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, trust_remote_code=True)


app = FastAPI(title="Local Qwen3 Embeddings", version="1.0")


@app.get("/health")
def health():
    model = get_model()
    return {"status": "ok", "model": MODEL_NAME, "dimension": model.get_sentence_embedding_dimension()}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts or any(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="input must contain non-empty text")
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"unsupported model: {request.model}")
    try:
        kwargs = {"normalize_embeddings": True, "show_progress_bar": False}
        if request.input_type == "query":
            kwargs["prompt_name"] = "query"
        vectors = get_model().encode(texts, **kwargs)
    except Exception as error:
        raise HTTPException(status_code=500, detail="embedding generation failed") from error
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector.tolist()}
            for index, vector in enumerate(vectors)
        ],
        "model": MODEL_NAME,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
