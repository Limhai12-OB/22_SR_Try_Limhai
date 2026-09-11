"""
Stage 2 of the pipeline: given a question, find the chunks most likely to
contain the answer.
"""
from typing import List, TypedDict

from app.config import TOP_K
from app.embeddings import embed_query
from app.vector_store import get_collection, search_vectors


class RetrievedChunk(TypedDict):
    text: str
    source: str
    chunk_index: int
    distance: float


def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    query_embedding = embed_query(query)

    results = search_vectors(collection, query_embedding, min(top_k, count))

    chunks: List[RetrievedChunk] = []
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for text, meta, distance in zip(documents, metadatas, distances):
        if text is None:
            continue
        meta = meta or {}
        chunks.append(
            {
                "text": text,
                "source": str(meta.get("source", "unknown")),
                "chunk_index": int(str(meta.get("chunk_index", -1))),
                "distance": distance,
            }
        )
    return chunks
