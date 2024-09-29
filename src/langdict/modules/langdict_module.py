from typing import Dict, Any

from langdict import LangDict

from .module import Module


class LangDictModule(Module):

    """LangDictModule: A module wrapper for LangDict."""

    def __init__(self, lang_dict: LangDict):
        super().__init__()
        self.lang_dict = lang_dict

    def __call__(
        self,
        *args,
        stream: bool = False,
        batch: bool = False,
        **kwargs
    ):
        if (
            self.streaming and
            self.is_last_child
        ):
            stream = True

        inputs = self.forward(*args, **kwargs)

        return self.lang_dict(
            inputs,
            stream=stream,
            batch=batch,
            trace_backend=self.trace_backend,
            module_name=self._get_name(),
        )

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        if type(args[0]) is dict:
            return args[0]
        else:
            raise ValueError("Invalid inputs type. Expected dict.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LangDictModule":
        return LangDictModule(LangDict.from_dict(data))

    def as_dict(self) -> Dict[str, Any]:
        return self.lang_dict.as_dict()
