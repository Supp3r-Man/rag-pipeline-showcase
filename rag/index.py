"""Embed chunks and search them with cosine similarity.

Anthropic doesn't offer an embeddings endpoint, so embeddings are computed
locally with fastembed (ONNX, no PyTorch needed). The "vector database" is a
numpy matrix on disk - plenty for thousands of chunks, and easy to read.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from .chunker import Chunk

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors  # (n_chunks, dim), L2-normalized
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(EMBED_MODEL)
        return self._model

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorIndex":
        index = cls(chunks, np.empty((0, 0)))
        embeddings = np.array(list(index.model.embed([c.text for c in chunks])))
        index.vectors = _normalize(embeddings)
        return index

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = _normalize(np.array(list(self.model.query_embed(query))))[0]
        scores = self.vectors @ q  # cosine similarity, since both sides are unit length
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy")
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in raw], vectors)
