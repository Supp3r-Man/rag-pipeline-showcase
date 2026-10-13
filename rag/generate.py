"""Answer a question from retrieved chunks using Claude."""

import os

import anthropic

from .index import Hit

MODEL = os.environ.get("RAG_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """\
You answer questions using only the numbered sources provided in <sources>.
Cite the sources you rely on inline like [1] or [2][3].
If the sources don't contain the answer, say you don't know - do not use outside knowledge."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f'<source id="{i}" file="{h.chunk.source}">\n{h.chunk.text}\n</source>'
        for i, h in enumerate(hits, start=1)
    )
    return f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"


def answer(question: str, hits: list[Hit], client: anthropic.Anthropic | None = None) -> str:
    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        # If a safety classifier declines the request, the API re-runs it on a
        # fallback model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this question."
    return "".join(block.text for block in response.content if block.type == "text")
