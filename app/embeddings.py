"""
One function, one job: turn text into vectors using the local Ollama
embedding model. nomic-embed-text
Both ingestion (embedding chunks) and retrieval
(embedding the user's question) call this, so the two are guaranteed
to use the same model and never drift apart.
"""
from typing import List

import ollama

from app.config import EMBED_MODEL


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of strings. Returns one vector per input string, same order."""
    if not texts:
        return []
    vectors: List[List[float]] = []
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        response = ollama.embed(model=EMBED_MODEL, input=batch)
        if len(response.embeddings) != len(batch) or any(not vector for vector in response.embeddings):
            raise RuntimeError("Ollama returned an incomplete embedding batch")
        vectors.extend([list(vector) for vector in response.embeddings])
    return vectors


def embed_query(text: str) -> List[float]:
    """Convenience wrapper for embedding a single piece of text (e.g. a question)."""
    return embed_texts([text])[0]
