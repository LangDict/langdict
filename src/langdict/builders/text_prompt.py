from typing import Union

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langdict.specs import TextPromptSpecification, ChatPromptSpecification

from .base import Builder


class PromptTemplateBuilder(Builder):
    """Prompt Template Builder interface"""

    def __init__(self):
        pass

    @classmethod
    def build(cls, spec: Union[TextPromptSpecification, ChatPromptSpecification]):
        if isinstance(spec, TextPromptSpecification):
            return PromptTemplate.from_template(spec.text)
        elif isinstance(spec, ChatPromptSpecification):
            return ChatPromptTemplate.from_messages(spec.messages)
        else:
            raise ValueError(f"Invalid specification type: {type(spec)}")

