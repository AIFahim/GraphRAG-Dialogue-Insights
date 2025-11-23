"""
Custom HuggingFace LLM wrapper for LangChain using InferenceClient
"""
from typing import Any, List, Optional
from langchain.llms.base import LLM
from huggingface_hub import InferenceClient


class HuggingFaceInferenceLLM(LLM):
    """Custom LLM wrapper for HuggingFace InferenceClient"""

    client: Any = None
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 2048

    def __init__(self, token: str, model: str = "Qwen/Qwen2.5-Coder-7B-Instruct", **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(token=token)
        self.model = model

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Call the HuggingFace InferenceClient"""

        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content.strip()

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
