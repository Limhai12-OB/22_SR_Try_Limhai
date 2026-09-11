# Local RAG App

This app answers questions using documents stored in the `data/` folder. It runs
locally with Ollama and retrieves relevant document chunks before generating an
answer.

## How to run

Install Python 3.11–3.14, [Poetry](https://python-poetry.org/), and
[Ollama](https://ollama.com/). Then run these commands from the project folder:

```bash
poetry install
ollama pull nomic-embed-text
ollama pull llama3.2:3b
poetry run python -m app.ingestion
poetry run python -m app.main
```

Make sure Ollama is running before starting the app. Enter a question at the
`You:` prompt, or enter `exit` to stop. To display the document chunks retrieved
for each answer, start the app with:

```bash
poetry run python -m app.main --show-chunks
```

Run the ingestion command again whenever documents in `data/` are added,
changed, or removed.

## Chunking strategy

The app uses fixed-size chunks of **800 characters** with a **120-character
overlap** between consecutive chunks. Fixed-size chunking is simple and keeps
each piece of context small enough for efficient retrieval. The overlap retains
context near chunk boundaries, reducing the chance that a sentence or important
instruction is split across two chunks.

## Embeddings and vector database

The embedding model is **`nomic-embed-text`**, run locally through Ollama. The
same model embeds both document chunks and user questions so their vectors can
be compared consistently.

The vector database is **ChromaDB**. It stores the document text, embeddings,
and source metadata persistently in the local `chroma_db/` directory. During a
question, the app searches ChromaDB for the most similar chunks and passes them
to the `llama3.2:3b` generation model to produce the answer.
