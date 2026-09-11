from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import generator as generate, ingestion as ingest, retriever as retrieval
from app import vector_store
from app.api import app


@pytest.mark.parametrize('size,overlap', [(0, 0), (-1, 0), (3, 3), (3, -1)])
def test_invalid_chunk_settings(size, overlap):
    with pytest.raises(ValueError):
        ingest.chunk_text('abcdef', size, overlap)


def test_chunk_overlap():
    assert ingest.chunk_text('abcdefgh', 4, 1) == ['abcd', 'defg', 'gh']


def test_empty_index_skips_embedding(monkeypatch):
    collection = Mock()
    collection.count.return_value = 0
    monkeypatch.setattr(retrieval, 'get_collection', lambda: collection)
    embed = Mock(side_effect=AssertionError('must not call Ollama'))
    monkeypatch.setattr(retrieval, 'embed_query', embed)
    assert retrieval.retrieve('question') == []


def test_retrieval_caps_count_and_handles_missing_metadata(monkeypatch):
    collection = Mock()
    collection.count.return_value = 1
    collection.query.return_value = {'documents': [['hello']], 'metadatas': [[None]], 'distances': [[0.2]]}
    monkeypatch.setattr(retrieval, 'get_collection', lambda: collection)
    monkeypatch.setattr(retrieval, 'embed_query', lambda _: [1.0])
    result = retrieval.retrieve('question', 100)
    assert collection.query.call_args.kwargs['n_results'] == 1
    assert result[0]['source'] == 'unknown'


def test_no_context_skips_generation(monkeypatch):
    monkeypatch.setattr(generate.ollama, 'chat', Mock(side_effect=AssertionError('must not generate')))
    assert 'enough information' in generate.generate_answer('question', [])


@pytest.mark.parametrize('top_k', [0, -1, True, 1.5])
def test_api_rejects_bad_top_k(top_k):
    assert TestClient(app).post('/chat', json={'question': 'hello', 'top_k': top_k}).status_code == 422


def test_api_rejects_blank_question():
    assert TestClient(app).post('/chat', json={'question': '  '}).status_code == 400


def test_failed_embedding_preserves_index(tmp_path, monkeypatch):
    (tmp_path / 'doc.txt').write_text('example', encoding='utf-8')
    monkeypatch.setattr(ingest, 'embed_texts', Mock(side_effect=ConnectionError('offline')))
    get_collection = Mock()
    monkeypatch.setattr(ingest, 'save_chunks', get_collection)
    with pytest.raises(ConnectionError):
        ingest.build_index(str(tmp_path))
    get_collection.assert_not_called()


def test_real_chroma_rebuild_removes_stale_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, 'CHROMA_DB_DIR', str(tmp_path / 'db'))
    monkeypatch.setattr(ingest, 'embed_texts', lambda texts: [[1.0, 0.0] for _ in texts])
    docs = tmp_path / 'docs'
    docs.mkdir()
    source = docs / 'first.txt'
    source.write_text('first document', encoding='utf-8')
    assert ingest.build_index(str(docs)) == 1
    source.unlink()
    (docs / 'second.txt').write_text('second document', encoding='utf-8')
    assert ingest.build_index(str(docs)) == 1
    assert vector_store.get_collection().get()['ids'] == ['second.txt::0']


def test_pipeline_passes_question_and_evidence(monkeypatch):
    from app import pipeline
    chunks = [{'text': 'evidence', 'source': 'doc.txt', 'chunk_index': 0, 'distance': 0.1}]
    retrieve = Mock(return_value=chunks)
    generate = Mock(return_value='grounded answer')
    monkeypatch.setattr(pipeline, 'retrieve', retrieve)
    monkeypatch.setattr(pipeline, 'generate_answer', generate)
    result = pipeline.ask('  question  ', 3)
    retrieve.assert_called_once_with('question', 3)
    generate.assert_called_once_with('question', chunks)
    assert result.sources == ['doc.txt']


def test_terminal_loop_handles_blank_question_and_exit(monkeypatch, capsys):
    from app import main
    from app.pipeline import Answer
    monkeypatch.setattr('sys.argv', ['chat'])
    monkeypatch.setattr(main, 'get_collection', lambda: Mock(count=lambda: 1))
    questions = iter(['', 'hello', 'EXIT'])
    monkeypatch.setattr('builtins.input', lambda _: next(questions))
    ask = Mock(return_value=Answer('answer', []))
    monkeypatch.setattr(main, 'ask', ask)
    main.main()
    ask.assert_called_once_with('hello')
    assert 'Assistant: answer' in capsys.readouterr().out


def test_terminal_empty_index_gives_setup_instruction(monkeypatch, capsys):
    from app import main
    monkeypatch.setattr('sys.argv', ['chat'])
    monkeypatch.setattr(main, 'get_collection', lambda: Mock(count=lambda: 0))
    main.main()
    assert 'app.ingestion' in capsys.readouterr().out
