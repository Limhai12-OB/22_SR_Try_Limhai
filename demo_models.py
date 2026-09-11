"""Standalone model checks, independent of the RAG pipeline."""
import ollama
from app.config import EMBED_MODEL, GEN_MODEL


def main():
    embedded = ollama.embed(model=EMBED_MODEL, input=['This is a test document.'])
    assert len(embedded.embeddings) == 1 and embedded.embeddings[0]
    print(f'{EMBED_MODEL}: OK, {len(embedded.embeddings[0])} dimensions')
    answer = ollama.chat(model=GEN_MODEL, messages=[{'role': 'user', 'content': 'Reply with only: Hello!'}], options={'temperature': 0})
    assert answer.message.content
    print(f'{GEN_MODEL}: {answer.message.content}')


if __name__ == '__main__':
    main()
