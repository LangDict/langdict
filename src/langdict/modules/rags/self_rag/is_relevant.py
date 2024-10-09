from typing import Any, Dict

from langdict import LangDict, LangDictModule


_SPECIFICATION = {
    "messages": [
        ("system", """You’ll be provided with an instruction, along with evidence and possibly some preceding sentences.
When there are preceding sentences, your focus should be on the sentence that comes after them.
Your job is to determine if the evidence is relevant to the initial instruction and the preceding context,
and provides useful information to complete the task described in the instruction.
If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].

## Demonstrations
- Instruction: Given four answer options, A, B, C, and D, choose the best answer.
- Input: Earth’s rotating causes
A: the cycling of AM and PM
B: the creation of volcanic eruptions
C: the cycling of the tides
D: the creation of gravity
- Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.
{{ "rating": "[Relevant]", "explanation": "The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A." }}

- Instruction: age to run for US House of Representatives
- Evidence: The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a senator represents at the time of election.
{{ "rating": "[Irrelevant]", "explanation": "The evidence only discusses the ages to run for the US Senate, not for the House of Representatives." }}

- Instruction: {instruction}
- Evidence: {evidence}
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


class IsRelevant(LangDictModule):
    """
    Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

    d provides useful information to solve x.

    Type: [IsRel]
    Output: {relevant, irrelevant}
    """

    def __init__(self):
        super().__init__(
            LangDict.from_dict(_SPECIFICATION)
        )

    def forward(self, instruction: str, evidence: str) -> Dict[str, Any]:
        return {
            "instruction": instruction,
            "evidence": evidence,
        }
