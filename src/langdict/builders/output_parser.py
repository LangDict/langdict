from enum import StrEnum

from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
    StrOutputParser,
)

from langdict.specs import OutputSpecification

from .base import Builder


class OutputType(StrEnum):
    STRING = "string"
    JSON = "json"


class OutputParserBuilder(Builder):

    def __init__(self):
        pass

    @classmethod
    def build(cls, spec: OutputSpecification) -> BaseOutputParser:
        if spec.type == OutputType.STRING.value:
            return StrOutputParser()
        elif spec.type == OutputType.JSON.value:
            return JsonOutputParser()
        else:
            raise ValueError("Invalid output parser type.")
