"""Terminal chat: poetry run python -m app.main."""
import utils.transport
import utils.session
import argparse
from ollama import ResponseError
from app.pipeline import ask
from app.vector_store import get_collection


def main() -> None:
    parser = argparse.ArgumentParser(description='Chat with your local documents')
    parser.add_argument('--show-chunks', action='store_true', help='Print retrieved evidence')
    args = parser.parse_args()
    if get_collection().count() == 0:
        print('No documents indexed. Run: poetry run python -m app.ingestion')
        return
    print('Local document chat. Type exit to quit. Each question is independent.')
    while True:
        try:
            question = input('\nYou: ').strip()
            if question.lower() == 'exit':
                break
            if not question:
                continue
            result = ask(question)
            if args.show_chunks:
                for chunk in result.chunks:
                    print(f"\n[{chunk['source']} chunk {chunk['chunk_index']}; distance={chunk['distance']:.4f}]\n{chunk['text']}")
            print(f'\nAssistant: {result.answer}')
            print('Retrieved sources: ' + ', '.join(result.sources))
        except (EOFError, KeyboardInterrupt):
            print('\nbyeeeee ')
            break
        except (ConnectionError, ResponseError) as exc:
            print(f'Ollama error: {exc}. Check ollama serve and the configured models.')


if __name__ == '__main__':
    main()
