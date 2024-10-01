from typing import Any, Dict

from langdict import LangDict, LangDictModule


_SPECIFICATION = {
    "messages": [
        ("system", """Compress the given text to short expressions, and such that you (GPT-4) can reconstruct it as close as possible to the original.
Unlike the usual text compression, I need you to comply with the 5 conditions below:
1. You can ONLY remove unimportant words.
2. Do not reorder the original words.
3. Do not change the original words.
4. Do not use abbreviations or emojis.
5. Do not add new words or symbols.
Compress the origin aggressively by removing words only.
Compress the origin as short as you can, while retaining as much information as possible.
If you understand, please compress the following text: {text}

The compressed text is:"""),
    ],
    "llm": {
        "model": "gpt-4o-mini",
    },
    "output": {
        "type": "string"
    },
    "metadata": {
        "arxiv": "https://arxiv.org/abs/2403.12968",
    }
}


class LLMLingua2(LangDictModule):

    def __init__(self):
        super().__init__(
            LangDict.from_dict(_SPECIFICATION)
        )

    def forward(self, text: str) -> Dict[str, Any]:
        return {
            "text": text
        }
