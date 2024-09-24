from typing import List, Tuple

from pydantic import BaseModel

from .base import Builder


class ChatPromptMessagesBuilder(Builder):

    def __init__(self):
        pass

    @classmethod
    def build(
        cls,
        persona: str = "",
        task_instruction: str = "",
        output_format: str = "",
        output_basemodel: BaseModel = None,
        fewshot_examples: List[str] = [],
        conversation_key: str = "conversation",
        context_key: str = "",
    ) -> List[Tuple[str, str]]:
        messages = []

        system_instruction = f"{persona}\n{task_instruction}"
        if output_format:
            system_instruction += f"\n## Output Format: {output_format}"
        elif output_basemodel:
            system_instruction += f"\n## Output Format: {output_basemodel.__name__}"

        if fewshot_examples:
            system_instruction += "\n## Examples:"
            for example in fewshot_examples:
                system_instruction += f"\n- {example}"

        messages.append(("system", system_instruction))

        if conversation_key:
            messages.append(("placeholder", "{" + conversation_key + "}"))
        if context_key:
            messages.append(("ai", "{" + context_key + "}"))
        return messages
