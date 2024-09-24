from typing import Dict, Optional

from .base import BaseSpecification


class LLMSpecification(BaseSpecification):

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 1,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        streaming: bool = False,
        n: int = 1,
        max_tokens: Optional[int] = None,
    ):
        super().__init__()

        self.model = model
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.streaming = streaming
        self.n = n
        self.max_tokens = max_tokens

    def validate(self):
        # TODO: 사용가능한 LLM 기준
        pass

    @classmethod
    def from_dict(cls, data: Dict) -> "LLMSpecification":
        return cls(
            model=data.get("model", "gpt-3.5-turbo"),
            model_name=None,
            # model_name=data.get("model_name", "openai"),
            api_key=data.get("api_key"),
            temperature=data.get("temperature", 1),
            top_p=data.get("top_p", None),
            top_k=data.get("top_k", None),
            streaming=data.get("streaming", False),
            n=data.get("n", 1),
            max_tokens=data.get("max_tokens", None),
        )

