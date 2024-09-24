from typing import Any, Dict



class BaseSpecification:

    def __init__(self):
        self.validate()

    def validate(self):
        raise NotImplementedError(
            f"validate method not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSpecification":
        raise NotImplementedError(
            f"from_dict method not implemented for {cls.__name__}"
        )

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__
