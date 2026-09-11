"""Offline pipeline: load, chunk, embed, and persist documents."""
import os
from typing import List, Tuple
from pypdf import PdfReader
from app.config import DATA_DIR, CHROMA_DB_DIR
from app.chunking import chunk_text
from app.embeddings import embed_texts
from app.vector_store import save_chunks

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_documents(data_dir: str = DATA_DIR) -> List[Tuple[str, str]]:
    """Return a list of (filename, full_text) for every .txt/.md/.pdf file in data_dir."""
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext in ("txt", "md"):
            text = _read_txt(path)
        elif ext == "pdf":
            text = _read_pdf(path)
        else:
            continue
        if text.strip():
            documents.append((filename, text))
    return documents


def build_index(data_dir: str = DATA_DIR) -> int:
    documents = load_documents(data_dir)
    if not documents:
        raise FileNotFoundError(f"No .txt/.md/.pdf files found in '{data_dir}/'")

    ids, texts, metadatas = [], [], []
    for filename, full_text in documents:
        for i, chunk in enumerate(chunk_text(full_text)):
            ids.append(f"{filename}::{i}")
            texts.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})

    embeddings = embed_texts(texts)
    save_chunks(ids, texts, embeddings, metadatas)
    return len(texts)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks from '{DATA_DIR}/' into Chroma at '{CHROMA_DB_DIR}/'.")
