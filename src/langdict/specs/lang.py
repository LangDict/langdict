from typing import Any, Dict

from .base import BaseSpecification
from .prompt import PromptSpecification
from .llm import LLMSpecification
from .output import OutputSpecification


class LangSpecification(BaseSpecification):

    REQUIRE_KEYS = [
        "llm",
        "output"
    ]

    def __init__(
        self,
        prompt: PromptSpecification,
        llm: LLMSpecification,
        output: OutputSpecification
    ):
        self.prompt = prompt
        self.llm = llm
        self.output = output

        super().__init__()

    def validate(self):
        self.prompt.validate()
        self.llm.validate()
        self.output.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LangSpecification":
        if any(key not in data for key in cls.REQUIRE_KEYS):
            raise ValueError(f"Missing keys in data. Required keys: {cls.REQUIRE_KEYS}")

        if (
            "text" in data and
            "messages" in data
        ):
            raise ValueError("Data cannot contain both 'text' and 'messages' keys.")

        prompt_type = None
        prompt_data = None
        if "text" in data:
            prompt_type = "text"
            prompt_data = data["text"]
        elif "messages" in data:
            prompt_type = "chat"
            prompt_data = data["messages"]
        else:
            raise ValueError("Data must contain either 'text' or 'messages' key.")

        prompt = PromptSpecification.from_dict(prompt_data, prompt_type=prompt_type)
        llm = LLMSpecification.from_dict(data["llm"])
        output = OutputSpecification.from_dict(data["output"])
        return cls(prompt, llm, output)

    def as_dict(self) -> Dict[str, Any]:
        data = self.prompt.as_dict()
        data["llm"] = self.llm.as_dict()
        data["output"] = self.output.as_dict()
        return data
