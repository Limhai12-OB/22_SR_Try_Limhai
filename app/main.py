"""
The whole app, end to end, exposed as a tiny API — no framework magic,
just three plain functions wired together.

Run it:
    python -m uvicorn app.main:app --reload

Then test it with curl (see README.md for full examples):
    curl -X POST http://127.0.0.1:8000/ingest
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
         -d '{"question": "What is the vacation policy?"}'
"""
import logging

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ollama import ResponseError
from fastapi.responses import JSONResponse

from app.generate import generate_answer
from app.ingest import build_index
from app.retrieval import retrieve

app = FastAPI(title="Baseline Chat-with-Documents API")


@app.exception_handler(ConnectionError)
async def ollama_unavailable(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Cannot connect to Ollama. Start it with ollama serve."})


@app.exception_handler(ResponseError)
async def ollama_error(request, exc):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, gt=0, strict=True)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ingest")
def ingest():
    """(Re)build the vector index from everything in data/."""
    try:
        count = build_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"chunks_indexed": count}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """The full retrieve -> augment -> generate loop for one question."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    chunks = retrieve(req.question, top_k=req.top_k) if req.top_k is not None else retrieve(req.question)
    answer = generate_answer(req.question, chunks)
    sources = sorted({c["source"] for c in chunks})
    return ChatResponse(answer=answer, sources=sources)
