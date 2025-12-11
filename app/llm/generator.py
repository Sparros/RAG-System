# app/llm/generator.py

from typing import List
from app.models.document_models import DocumentChunk


class LLMGenerator:
    """
    Handles prompt construction and calling the underlying LLM client.
    Expects the LLM client to have a `.complete(prompt: str) -> str` method.
    """

    def __init__(self, client):
        self.client = client

    def build_context(self, chunks: List[DocumentChunk], max_chars: int = 4000) -> str:
        """
        Build the reference context shown to the LLM.
        """
        parts = []
        length = 0

        for chunk in chunks:
            header = f"[SOURCE: {chunk.metadata.source}, page {chunk.metadata.page}]\n"
            body = chunk.text.strip() + "\n\n"
            block = header + body

            if length + len(block) > max_chars:
                break

            parts.append(block)
            length += len(block)

        return "".join(parts)

    def build_prompt(self, query: str, context: str) -> str:
        """
        The final template sent to the LLM.
        """
        return f"""
Use only the following context to answer the question.
If the answer is not explicitly stated, respond with:
"I don't know based on the provided context."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()

    def generate_answer(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Build context + prompt → send to underlying LLM.
        """
        context = self.build_context(chunks)
        prompt = self.build_prompt(query, context)

        return self.client.complete(prompt)
