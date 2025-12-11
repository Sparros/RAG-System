import requests

class OllamaClient:
    def __init__(self, model: str = "phi3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def complete(self, prompt: str) -> str:
        """
        Sends a prompt to the local Ollama model.
        Returns the response text.
        """
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
