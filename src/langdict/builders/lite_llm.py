
from langdict.chat_models import ChatLiteLLM
from langdict.specs import LLMSpecification

from .base import Builder


class LiteLLMBuilder(Builder):

    def __init__(self):
        pass

    @classmethod
    def build(cls, spec: LLMSpecification):
        return ChatLiteLLM(
            model=spec.model,
            model_name=spec.model_name,
            openai_api_key=spec.api_key,
            temperature=spec.temperature,
            top_p=spec.top_p,
            top_k=spec.top_k,
            streaming=spec.streaming,
            n=spec.n,
            max_tokens=spec.max_tokens,
        )
