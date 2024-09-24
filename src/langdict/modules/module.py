import json
from typing import Any, Dict, List, Optional, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda

from langdict.builders import TraceCallbackBuilder

from .parameter import Parameter


T = TypeVar("T", bound="Module")


class Module:

    """Module: base class for compound ai systems.

    Example:

        class RAG(Module):

        def __init__(self, docs: List[str]):
            super().__init__()
            self.query_rewrite = LangDictModule.from_dict({ ... })
            self.search = SimpleKeywordSearch(docs=docs)  # Module
            self.answer = LangDictModule.from_dict({ ... })

        def forward(self, inputs: Dict):
            query_rewrite_result = self.query_rewrite({
                "conversation": inputs["conversation"],
            })
            doc = self.search(query_rewrite_result)
            return self.answer({
                "conversation": inputs["conversation"],
                "context": doc,
            })

    """

    NAME: str = None

    batching: bool = False
    streaming: bool = False
    is_last_child: Optional["Module"] = None
    trace_backend: Optional[str] = None  # If None, no tracing

    def __init__(self):
        self.trace_backend = None
        self._parameters = {}
        self._modules = {}

    def __call__(
        self,
        *args,
        stream: bool = False,
        **kwargs,
    ):
        if (
            stream and
            self.is_last_child is None
        ):
            self.stream(stream)
            self._set_last_child()

        chain = RunnableLambda(lambda x: self.forward(x))
        callbacks = self._trace_callbacks(self.trace_backend, self._get_name())
        return chain.invoke(
            *args,
            config={"callbacks": callbacks},
            **kwargs,
        )

    def _trace_callbacks(
        self,
        trace_backend: str,
        module_name: str
    ) -> List[BaseCallbackHandler]:
        callbacks = []
        if trace_backend:
            builder = TraceCallbackBuilder()
            callback = builder.build(
                trace_backend,
                module_name=module_name,
            )
            callbacks.append(callback)
        return callbacks

    def __getattr__(self, name: str) -> "Module":
        if (
            "_parameters" not in self.__dict__ or
            "_modules" not in self.__dict__
        ):
            raise ValueError("Intialize Module first.")

        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: "Module") -> None:
        if isinstance(value, Module):
            value.NAME = name
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        else:
            object.__setattr__(self, name, value)

    def _get_name(self):
        if self.NAME:
            return self.NAME
        return self.__class__.__name__

    def forward(self):
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required \"forward\" function"
        )

    def children(self):
        """Yield all children modules."""
        visited = set()
        for name, module in self._modules.items():
            if module and module not in visited:
                visited.add(module)
                yield module

    def trace(self, backend: str = "console") -> T:
        """Set the trace backend for all modules.

        Args:
            backend (str): The trace backend to use. Default is "console".

        Returns:
            Module: self
        """
        self.trace_backend = backend

        for module in self.children():
            module.trace(backend)
        return self

    def stream(self, is_stream: bool = False) -> T:
        """Set the streaming flag for all modules.

        Args:
            is_stream (bool): The streaming flag to use. Default is False.

        Returns:
            Module: self
        """
        self.streaming = is_stream

        for module in self.children():
            module.stream(is_stream)
        return self

    def _set_last_child(self) -> "Module":
        modules = list(self.children())
        for m in modules:
            m.is_last_child = False
        last_child = modules[-1]

        while modules:
            modules = list(last_child.children())
            for m in modules:
                m.is_last_child = False
            if not modules:
                break
            last_child = modules[-1]

        last_child.is_last_child = True

    def save_json(self, filename: str) -> None:
        """Save the module as a json file.

        Args:
            filename (str): The filename to save the module to.
        """
        all_parameters = {}
        for name, param in self._parameters.items():
            all_parameters[name] = param.value

        all_modules = {}
        for module in self.children():
            all_modules[module.NAME] = module.as_dict()

        data = {
            "parameters": all_parameters,
            "modules": all_modules,
        }

        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_json(self, filename: str) -> T:
        """Load the module from a json file.

        Args:
            filename (str): The filename to load the module from.

        Returns:
            Module: self
        """
        with open(filename, "r") as f:
            data = json.load(f)

        parameter_data = data.get("parameters", {})
        module_data = data.get("modules", {})

        for name, value in parameter_data.items():
            self.__setattr__(name, Parameter(value))

        for module in self.children():
            langdict_data = module_data.get(module.NAME)
            if not langdict_data:
                continue
            self.__setattr__(module.NAME, module.__class__.from_dict(langdict_data))
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Module":
        return cls()

    def as_dict(self) -> Dict[str, Any]:
        return {}
