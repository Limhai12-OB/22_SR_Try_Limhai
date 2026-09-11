"""Online pipeline: question -> retrieval -> prompt augmentation -> answer."""
from dataclasses import dataclass
from app.config import TOP_K
from app.retriever import RetrievedChunk, retrieve
from app.generator import generate_answer


@dataclass
class Answer:
    answer: str
    chunks: list[RetrievedChunk]

    @property
    def sources(self) -> list[str]:
        return sorted({chunk['source'] for chunk in self.chunks})


def ask(question: str, top_k: int | None = None) -> Answer:
    question = question.strip()
    if not question:
        raise ValueError('question must not be empty')
    chunks = retrieve(question, TOP_K if top_k is None else top_k)
    return Answer(generate_answer(question, chunks), chunks)
