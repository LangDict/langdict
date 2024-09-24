from typing import Dict

from .base import BaseSpecification


class OutputSpecification(BaseSpecification):

    OUTPUT_TYPES = {"string", "json"}

    def __init__(self, type: str = "string"):
        self.type = type

        super().__init__()

    def validate(self):
        if self.type not in self.OUTPUT_TYPES:
            raise ValueError(f"Invalid output type: {self.type}")

    @classmethod
    def from_dict(cls, data: Dict) -> "OutputSpecification":
        return cls(
            type=data.get("type", "string"),
        )
