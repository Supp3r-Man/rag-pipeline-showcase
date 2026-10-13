# RAG Pipeline Showcase

A minimal retrieval-augmented generation (RAG) pipeline in ~150 lines of Python.
It answers questions about a document set using semantic search for retrieval and
Claude for grounded, cited answers.

```
 docs/*.md ──► chunk ──► embed (local) ──► index/ (numpy)
                                                │
 question ──► embed ──► cosine top-k ◄──────────┘
                              │
                              ▼
            numbered <source> blocks + question ──► Claude ──► cited answer
```

| Stage | File | What it does |
|---|---|---|
| Chunk | `rag/chunker.py` | Paragraph-aligned chunks (~800 chars) with one paragraph of overlap |
| Embed + index | `rag/index.py` | Local embeddings (`bge-small-en-v1.5` via [fastembed](https://github.com/qdrant/fastembed)), stored as a normalized numpy matrix, searched by cosine similarity |
| Generate | `rag/generate.py` | Puts the top-k chunks in the prompt and asks Claude to answer only from them, citing `[1]`, `[2]`... |

The sample corpus in `data/docs/` is a **fictional** company handbook, so a correct
answer can only come from retrieval, not from the model's training data.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                  # Windows: $env:ANTHROPIC_API_KEY="..."

python ingest.py                                     # build the index (downloads a ~70 MB embedding model once)
python ask.py "How many PTO days can I carry over?"
python ask.py "What's the hotel cap per night?" -k 3
python ask.py "What is Nimbus's parental leave policy?"   # not in the docs -> should say it doesn't know
python ask.py "..." --retrieve-only                  # inspect retrieval without calling the LLM
```

Use your own documents by pointing at another folder: `python ingest.py --docs path/to/docs`
(`.md` and `.txt` files are read).

The answer model defaults to `claude-opus-5-5`; override it with `RAG_MODEL=<model id>`.

## Tests

```bash
pytest
```

Covers chunking behavior, retrieval recall@3 on four questions, index save/load, and
prompt construction. The tests don't call the Claude API.

## Design notes

- **Why local embeddings?** Anthropic doesn't offer an embeddings endpoint, so retrieval runs
  on-device. The generation step is the only network call.
- **Why numpy instead of a vector database?** For thousands of chunks, a matrix multiply is
  fast and keeps the code readable. Swap `VectorIndex` for FAISS, Chroma, or pgvector when you
  outgrow it; the interface is just `build`, `search`, `save`, `load`.
- **Grounding:** the system prompt restricts answers to the provided sources and tells the
  model to say "I don't know" otherwise.
- **Refusals:** the request opts into server-side fallbacks (`fallbacks="default"`), and a
  `refusal` stop reason is handled before reading the response.

## Known limitations / next steps

- Small embedding models miss some paraphrases (e.g. "vacation days" vs "PTO"). That's why the
  tests measure recall@3 rather than top-1. A reranker or hybrid BM25 + vector search would help.
- Fixed-size paragraph chunking ignores document structure; heading-aware splitting is a natural upgrade.
- No evaluation harness beyond the retrieval tests. A set of question/answer pairs with an
  LLM judge would be the next addition.
