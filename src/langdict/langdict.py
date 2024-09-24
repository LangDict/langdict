from typing import Any, Dict, List, Tuple, Union

from langchain_core.callbacks import BaseCallbackHandler

from langdict.specs import LangSpecification
from langdict.builders import (
    PromptTemplateBuilder,
    LiteLLMBuilder,
    OutputParserBuilder,
    TraceCallbackBuilder,
)


class LangDict:

    """LangDict: A unit of simple llm chain.

    Chain Structure:
        [Prompt] -> [LLM] -> [Output Parser]
    """

    def __init__(self, spec: LangSpecification):
        self.spec = spec

        prompt = PromptTemplateBuilder.build(spec.prompt)
        llm = LiteLLMBuilder.build(spec.llm)
        output_parser = OutputParserBuilder.build(spec.output)

        chain = prompt | llm | output_parser
        self.chain = chain

    def __call__(
        self,
        inputs: Union[
            Dict[str, Any], List[Tuple[str, Dict[str, Any]]]
        ],
        stream: bool = False,
        batch: bool = False,
        trace_backend: str = None,
        module_name: str = None,
    ):
        """Invoke the chain with inputs.

        Example::

            chitchat({
                "conversation": [("user", "Hello, how are you doing?")]
            })
            chitchat({
                "conversation": [("user", "Hello, how are you doing?")]
            }, stream=True)
            chitchat([inputs, inputs], batch=True)

        Args:
            inputs: input data for the chain.
            stream: enable streaming mode.
            batch: enable batch mode.
            trace_backend: trace backend to use. if None, no tracing.
            module_name: name of the module for tracing.

        """

        # TODO: async implementation

        callbacks = self._trace_callbacks(trace_backend, module_name)

        if isinstance(inputs, dict):
            if stream:
                return self.chain.stream(
                    inputs,
                    config={"callbacks": callbacks}
                )
            else:
                return self.chain.invoke(
                    inputs,
                    config={"callbacks": callbacks}
                )
        elif isinstance(inputs, list):
            if batch:
                return self.chain.batch(
                    inputs,
                    config={"callbacks": callbacks}
                )
            else:
                raise ValueError("List inputs must be batched.")
        else:
            raise ValueError("Invalid inputs type.")

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LangDict":
        """Create LangDict from dictionary data.

        Example::

            chitchat = LangDict.from_dict({
                "prompt": {
                    "type": "chat",
                    "messages": [
                        ("system", "You are a helpful AI bot. Your name is {name}."),
                        ("human", "Hello, how are you doing?"),
                        ("ai", "I'm doing well, thanks!"),
                        ("human", "{user_input}"),
                    ]
                },
                "llm": {
                    "model": "gpt-4o",
                    "max_tokens": 200
                },
                "output": {
                    "type": "string"
                }
            })

        Args:
            data: specification data for the LangDict
                (must include ('text' or 'messages'), 'llm', 'output' keys)
        """

        lang_spec = LangSpecification.from_dict(data)
        return LangDict(lang_spec)

    def as_dict(self) -> Dict[str, Any]:
        return self.spec.as_dict()
