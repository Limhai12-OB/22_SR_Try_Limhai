"""Record five actual local RAG runs, including complete retrieved chunks."""
import json
from datetime import datetime, timezone
from pathlib import Path
from app.config import EMBED_MODEL, GEN_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from app.pipeline import ask

QUESTIONS = [
    'What should I do first when my printer is jammed?',
    'How do I configure VPN access for remote work?',
    'How do I back up important files?',
    'How do I secure my wireless network?',
    'What is the company vacation policy?',
]


def main():
    folder = Path(__file__).resolve().parent / 'reports'
    folder.mkdir(exist_ok=True)
    runs = []
    for question in QUESTIONS:
        result = ask(question)
        runs.append({'question': question, 'chunks': result.chunks, 'answer': result.answer})
        print(f'{len(runs)}/5: {question}', flush=True)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'generation_model': GEN_MODEL,
              'embedding_model': EMBED_MODEL, 'chunk_size': CHUNK_SIZE,
              'chunk_overlap': CHUNK_OVERLAP, 'top_k': TOP_K, 'runs': runs}
    (folder / 'test_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Actual local RAG test results', '', f"Generated: {report['created_at']}",
             '', f'Models: {EMBED_MODEL} / {GEN_MODEL}; chunks: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}, top_k: {TOP_K}.']
    for i, run in enumerate(runs, 1):
        lines += ['', f"## {i}. {run['question']}", '', '### Retrieved chunks']
        for c in run['chunks']:
            lines += ['', f"**{c['source']} — chunk {c['chunk_index']}, distance {c['distance']:.4f}**", '', '```text', c['text'], '```']
        lines += ['', '### Answer', '', run['answer']]
    (folder / 'test_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
