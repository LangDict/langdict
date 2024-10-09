from typing import Any, Dict

from langdict import LangDict, LangDictModule


_SPECIFICATION = {
    "messages": [
        ("system", """Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest).
We call this score perceived utility. The detailed criterion is as follows:

5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.
4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence.
3: The response is acceptable, but some major additions or improvements are needed to satisfy users’ needs.
2: The response still addresses the main request, but it is not complete or not relevant to the query.
1: The response is barely on-topic or completely irrelevant.

## Demonstrations
- Instruction: Who is the current prime minister of the UK as of 2023?
- Output: Boris Johnson was the prime minister of the UK from 2019 - 2022.
{{ "utility": 2, "explanation": "While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn’t answer the instruction. Therefore, the utility is 2." }}

- Instruction: Given a description of a travel destination, recommend 10 tourist attractions with detailed descriptions of each. The travel destination is Tokyo, Japan.
- Output: Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen.
{{ "utility": 3, "explanation": "This output doesn’t provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn’t match the instructions strictly." }}

- Instruction: {instruction}
- Output: {output}
"""),
    ],
    "llm": {
        "model": "gpt-4o-mini",
    },
    "output": {
        "type": "json"
    },
    "metadata": {
        "arxiv": "https://arxiv.org/abs/2310.11511",
    }
}


class IsUseful(LangDictModule):
    """
    Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

    y is a useful response to x.

    Type: [IsUse]
    Output: {5, 4, 3, 2, 1}
    """

    def __init__(self):
        super().__init__(
            LangDict.from_dict(_SPECIFICATION)
        )

    def forward(
        self,
        instruction: str,
        preceding: str,
        output: str,
        evidence: str,
    ) -> Dict[str, Any]:
        return {
            "instruction": instruction,
            "preceding": preceding,
            "output": output,
            "evidence": evidence,
        }
