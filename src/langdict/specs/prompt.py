from typing import Any, Dict, List, Tuple, Union

from .base import BaseSpecification


class PromptSpecification(BaseSpecification):

    def __init__(self):
        self.validate()

    def validate(self):
        raise NotImplementedError("PromptSpecification.validate() must be implemented in a subclass.")

    @classmethod
    def from_dict(
        cls,
        data: Union[Dict, List],
        prompt_type: str = "chat"
    ) -> Union["TextPromptSpecification", "ChatPromptSpecification"]:
        if prompt_type == "text":
            return TextPromptSpecification(data)
        elif prompt_type == "chat":
            return ChatPromptSpecification.from_dict(data)
        else:
            raise ValueError("Invalid prompt type.")


class TextPromptSpecification(PromptSpecification):

    def __init__(self, text: str):
        self.text = text

        super().__init__()

    def validate(self):
        if not self.text:
            raise ValueError("Text prompt is empty.")
        if (
            "{" not in self.text or
            "}" not in self.text
        ):
            raise ValueError("PromptSpecification is missing placeholders.")


class ChatPromptSpecification(PromptSpecification):

    def __init__(self, messages: List[Tuple[str, str]]):
        self.messages = messages

        super().__init__()

    def validate(self):
        required_keys = {"human", "ai", "system", "placeholder"}

        for m in self.messages:
            if m[0] not in required_keys:
                raise ValueError(f"Invalid role in message: {m[0]}")

    @classmethod
    def from_dict(cls, data: List[Union[List[str], Tuple[str, str]]]) -> "ChatPromptSpecification":
        messages = []
        for d in data:
            if type(d) == list:
                messages.append(tuple(d))
            else:
                messages.append(d)
        return cls(messages)
