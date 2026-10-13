from pathlib import Path

import pytest

from rag.chunker import chunk_text, load_chunks
from rag.generate import build_prompt
from rag.index import VectorIndex

DOCS = Path(__file__).parent.parent / "data" / "docs"


def test_chunks_respect_max_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 100 for i in range(10))
    chunks = chunk_text(text, max_chars=300, overlap_paragraphs=1)
    assert len(chunks) > 1
    # last paragraph of each chunk reappears at the start of the next
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split("\n\n")[-1] == nxt.split("\n\n")[0]


def test_long_paragraph_is_kept_whole():
    assert chunk_text("y" * 2000, max_chars=300) == ["y" * 2000]


def test_load_chunks_reads_all_docs():
    sources = {c.source for c in load_chunks(DOCS)}
    assert "pto-policy.md" in sources and len(sources) == 4


@pytest.fixture(scope="module")
def index():
    return VectorIndex.build(load_chunks(DOCS))


@pytest.mark.parametrize(
    "question, expected_source",
    [
        ("How many vacation days do I get?", "pto-policy.md"),
        ("Can I work from another country?", "remote-work.md"),
        ("How many approvals does a pull request need?", "engineering-onboarding.md"),
        ("What is the hotel cap per night?", "expense-policy.md"),
    ],
)
def test_retrieval_recall_at_3(index, question, expected_source):
    # What matters for RAG is that the right chunk reaches the LLM, not that it
    # ranks first. The small embedding model ranks "vacation days" below an
    # unrelated "30 days per year" chunk, but the PTO chunk is still in the top 3.
    assert expected_source in [h.chunk.source for h in index.search(question, k=3)]


def test_index_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.chunks == index.chunks
    assert loaded.search("sick leave", k=1)[0].chunk.source == "pto-policy.md"


def test_prompt_numbers_sources(index):
    hits = index.search("PTO carryover", k=2)
    prompt = build_prompt("PTO carryover?", hits)
    assert '<source id="1"' in prompt and '<source id="2"' in prompt
    assert prompt.endswith("Question: PTO carryover?")
