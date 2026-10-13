"""Split documents into overlapping, paragraph-aligned chunks."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    source: str  # file name the chunk came from
    index: int  # position of the chunk within that file
    text: str


def chunk_text(text: str, max_chars: int = 800, overlap_paragraphs: int = 1) -> list[str]:
    """Pack paragraphs into chunks of at most ~max_chars.

    The last `overlap_paragraphs` paragraphs of each chunk are repeated at the
    start of the next one so facts that straddle a boundary stay retrievable.
    A single paragraph longer than max_chars becomes its own chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for para in paragraphs:
        if current and size + len(para) > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            size = sum(len(p) for p in current)
        current.append(para)
        size += len(para)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def load_chunks(docs_dir: str | Path, **kwargs) -> list[Chunk]:
    """Read every .md/.txt file under docs_dir and chunk it."""
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(chunk_text(text, **kwargs)):
            chunks.append(Chunk(source=path.name, index=i, text=piece))
    return chunks
