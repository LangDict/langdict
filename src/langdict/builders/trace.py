from typing import List

from langdict.traces import (
    TraceBackend,
    TraceStdOutCallbackHandler,
)

from .base import Builder


class TraceCallbackBuilder(Builder):

    def __init__(self):
        pass

    def build(
        self,
        backend: str,
        module_name: str = None,
        tags: List[str] = None,
        session_id: str = None,
        user_id: str = None,
    ):

        if backend == TraceBackend.CONSOLE:
            return TraceStdOutCallbackHandler(module_name=module_name)
        elif backend == TraceBackend.LANGFUSE:
            try:
                from langfuse.callback import CallbackHandler
            except ImportError:
                raise ModuleNotFoundError("LangFuse is not installed.")

            return CallbackHandler(
                trace_name=module_name,
                tags=tags,
                session_id=session_id,
                user_id=user_id,
            )
        elif backend == TraceBackend.LANGSMITH:
            try:
                from langchain_core.tracers import LangChainTracer
            except ImportError:
                raise ModuleNotFoundError("LangChainTracer is not installed.")

            tags.append(module_name)

            return LangChainTracer(
                example_id=session_id,
                tags=tags,
            )
        else:
            raise ValueError(f"Backend {backend} is not supported")
