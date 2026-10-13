"""Chunk the documents in data/docs, embed them, and save the index to ./index."""

import argparse

from rag.chunker import load_chunks
from rag.index import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs")
    parser.add_argument("--out", default="index")
    args = parser.parse_args()

    chunks = load_chunks(args.docs)
    print(f"Chunked {len({c.source for c in chunks})} files into {len(chunks)} chunks")
    VectorIndex.build(chunks).save(args.out)
    print(f"Index saved to {args.out}/")


if __name__ == "__main__":
    main()
