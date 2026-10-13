"""Ask a question: retrieve the top chunks, then have Claude answer from them."""

import argparse

from rag.generate import answer
from rag.index import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=4, help="chunks to retrieve")
    parser.add_argument("--index", default="index")
    parser.add_argument(
        "--retrieve-only", action="store_true", help="show retrieved chunks, skip the LLM call"
    )
    args = parser.parse_args()

    hits = VectorIndex.load(args.index).search(args.question, k=args.k)

    print("Retrieved:")
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] {h.chunk.source}#{h.chunk.index}  score={h.score:.3f}")

    if not args.retrieve_only:
        print(f"\n{answer(args.question, hits)}")


if __name__ == "__main__":
    main()
