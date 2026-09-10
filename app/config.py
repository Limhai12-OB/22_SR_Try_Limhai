"""
All the knobs students can turn are collected here, in one place.
Change a value here and the whole app picks it up — no hunting through files.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Models (must already be pulled in Ollama: `ollama pull <name>`) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text").strip()
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2:3b").strip()

# --- Storage ---
DATA_DIR = str(PROJECT_ROOT / os.getenv("DATA_DIR", "data"))
CHROMA_DB_DIR = str(PROJECT_ROOT / os.getenv("CHROMA_DB_DIR", "chroma_db"))
COLLECTION_NAME = "documents"

# --- Chunking ---
CHUNK_SIZE = 800     # characters per chunk (not tokens — good enough for a baseline)
CHUNK_OVERLAP = 120  # characters shared between consecutive chunks

# --- Retrieval ---
TOP_K = 4            # how many chunks to pull back per question

# --- Generation ---
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer is not contained in the context, "
    "say \"I don't have enough information in the documents to answer that.\" "
    "Do not use outside knowledge. Cite the source file name(s) you used."
)
