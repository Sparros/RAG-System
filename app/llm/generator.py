from typing import List
from app.models.document_models import DocumentChunk


class LLMGenerator:
    """
    Simple wrapper around an LLM client.
    This class is responsible for:
    - building the final prompt
    - calling the LLM
    """

    def __init__(self, client):
        """
        `client` can be an OpenAI client or any other LLM client
        with a `.complete(prompt: str) -> str` interface.
        """
        self.client = client

    def build_context(self, chunks: List[DocumentChunk], max_chars: int = 4000) -> str:
        """
        Build a context string from retrieved chunks with basic truncation.
        """
        parts: List[str] = []
        current_len = 0

        for chunk in chunks:
            header = f"[SOURCE: {chunk.metadata.source}, page {chunk.metadata.page}]\n"
            text = chunk.text.strip() + "\n\n"
            piece = header + text

            if current_len + len(piece) > max_chars:
                break

            parts.append(piece)
            current_len += len(piece)

        return "".join(parts)

    def build_prompt(self, query: str, context: str) -> str:
        """
        Build the final prompt for the LLM.
        """
        prompt = f"""You are a helpful assistant. Use ONLY the information in the context below to answer the question.

If the answer cannot be found in the context, say "I don't know based on the provided context."

CONTEXT:
{context}

QUESTION:
{query}

Answer:
"""
        return prompt

    def generate_answer(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Main entry: build context, prompt, and call the LLM client.
        """
        context = self.build_context(chunks)
        prompt = self.build_prompt(query, context)

        # Expect client to have a `.complete(prompt)` method.
        response_text = self.client.complete(prompt)

        return response_text
