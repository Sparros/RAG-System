import pytest
from app.llm.ollama_client import OllamaClient
import requests

def ollama_available():
    """Check if Ollama server is running locally."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama is not running or model is unavailable"
)
def test_ollama_basic_completion():
    client = OllamaClient(model="phi3")
    prompt = "Explain what a neural network is in one sentence."

    answer = client.complete(prompt)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "network" in answer.lower() or "neur" in answer.lower()
