# Chat with local documents

A FastAPI application that indexes TXT, Markdown, and text-based PDF files in
Chroma, retrieves relevant chunks with Ollama embeddings, and generates answers
with source filenames using a local Ollama model.

## Setup

Use a Python virtual environment (the current project uses Python 3.14):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Ollama must be running (`ollama serve` if it is not already running as a service).
Put documents in `data/`, then run from the project root:

```bash
python -m app.ingest
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API, or use:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I reset a jammed printer?","top_k":4}'
curl -X POST http://127.0.0.1:8000/ingest
```

`POST /ingest` updates the index and removes chunks for deleted documents after
successful writes. Failed loading or embedding preserves the previous index.
Multiple writes are not a transaction; do not run simultaneous ingestion jobs.
An empty index returns an insufficient-information answer without calling Ollama.

## Configuration

Defaults live in `app/config.py`. Export `EMBED_MODEL`, `GEN_MODEL`, `DATA_DIR`,
`CHROMA_DB_DIR`, or `OLLAMA_HOST` to override them. Relative data/storage paths
resolve against the project root. The default generation model is `llama3.2:3b`,
which is already installed on this machine. Environment files are not loaded automatically.

When changing embedding models, use a new `CHROMA_DB_DIR` and ingest again;
vectors from different embedding models cannot be mixed. PDFs must contain
extractable text; OCR is not included. Retrieval returns nearest chunks and does
not guarantee that a document contains the answer.

## Checks

```bash
python -m pytest -q
python -m pip check
```

Tests use temporary Chroma storage and mocked model calls; they do not require
Ollama or modify the normal index. Bind the development server to localhost.
# 22_SR_Try_Limhai
