"""Persistent Chroma storage; all vectors are supplied by our Ollama code."""
import chromadb
from chromadb.config import Settings
from app.config import CHROMA_DB_DIR, COLLECTION_NAME

_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=_CHROMA_SETTINGS)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=None)


def save_chunks(ids, texts, embeddings, metadatas):
    collection = get_collection()
    old_ids = set(collection.get()["ids"])
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end], documents=texts[start:end],
            embeddings=embeddings[start:end], metadatas=metadatas[start:end],
        )
    stale_ids = sorted(old_ids - set(ids))
    for start in range(0, len(stale_ids), batch_size):
        collection.delete(ids=stale_ids[start:start + batch_size])

def search_vectors(collection, vector, top_k):
    return collection.query(query_embeddings=[vector], n_results=top_k,
                            include=["documents", "metadatas", "distances"])
