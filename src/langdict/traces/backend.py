from enum import StrEnum


class TraceBackend(StrEnum):
    CONSOLE = "console"
    LANGFUSE = "langfuse"
