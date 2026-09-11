"""Search the vector store directly, without generation."""
import argparse
from app.embeddings import embed_query
from app.vector_store import get_collection, search_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', nargs='?', default='How do I reset a jammed printer?')
    args = parser.parse_args()
    collection = get_collection()
    if not collection.count():
        raise SystemExit('Index is empty. Run poetry run python -m app.ingestion first.')
    results = search_vectors(collection, embed_query(args.question), min(3, collection.count()))
    for i, (text, meta, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1):
        print(f"\n{i}. {meta['source']} chunk {meta['chunk_index']} (distance {distance:.4f})\n{text}")


if __name__ == '__main__':
    main()
