from typing import Any, Dict, List

from langdict import LangDict, LangDictModule, Module

from .need_retrieve import NeedRetrieve
from .is_relevant import IsRelevant
from .is_support import IsSupport
from .is_useful import IsUseful


_GENERATE_SPECIFICATION = {
    "messages": [
        ("system", """Given an instruction and evidence, please make a answer.
- Instruction: {instruction}
- Evidence: {evidence}
{preceding}
"""),
    ],
    "llm": {
        "model": "gpt-4o-mini",
    },
    "output": {
        "type": "string"
    },
}


class SelfRAG(Module):
    """
    Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
    """

    def __init__(self, retriever: "Retriever"):
        super().__init__()
        self.retriever = retriever
        self.segment_generator = LangDictModule(
            LangDict.from_dict(_GENERATE_SPECIFICATION)
        )

        # Reflection token
        self.retrieve = NeedRetrieve()
        self.is_rel = IsRelevant()
        self.is_sup = IsSupport()
        self.is_use = IsUseful()

    def forward(
        self,
        instruction: str,
        preceding: str = "",
        output: str = "",
        evidence: str = "",
    ) -> str:
        # Step 1: Retrieve on demand
        retrieve_token = self.retrieve(instruction)
        if retrieve_token == "[yes]":
            passages = self.retriever.search(instruction, preceding)

            inputs = []
            for i, passage in enumerate(passages):
                inputs.append({
                    "index": i,
                    "instruction": instruction,
                    "preceding": preceding,
                    "evidence": passage,
                    "output": output,
                })
            is_relevants = self.is_rel(inputs, batch=True)

            filtered_inputs = []
            for i, is_relevant in enumerate(is_relevants):
                if is_relevant == "[Relevant]":
                    filtered_inputs.append(inputs[i])
            generated_segments = self.segment_generator(filtered_inputs, batch=True)
            for i, segment in enumerate(generated_segments):
                filtered_inputs[i]["output"] = segment

            is_supports = self.is_sup(filtered_inputs, batch=True)
            is_usefuls = self.is_use(filtered_inputs, batch=True)

            for input, is_relevant, is_support, is_useful in zip(filtered_inputs, is_relevants, is_supports, is_usefuls):
                index = input["index"]
                inputs[index].update({
                    "is_relevant": is_relevant,
                    "is_support": is_support,
                    "is_useful": is_useful,
                })

            ranking = self.ranking(inputs)
            return generated_segments[ranking[0]]
        elif (
            retrieve_token == "[continue]" or
            retrieve_token == "[no]"
        ):
            return self.segment_generator([{
                "instruction": instruction,
                "preceding": preceding,
                "evidence": evidence
            }])
        else:
            raise ValueError(f"Invalid retrieve token: {retrieve_token}")

    def ranking(self, segments: List[Dict[str, Any]]) -> List[int]:
        """ Heuristic ranking based on relevance, support, and usefulness. """

        filtered_segments = [segment for segment in segments if segment["is_relevant"] == "[Relevant]"]
        sorted_segments = sorted(filtered_segments, key=lambda x: (x["is_support"], -x["is_useful"]))
        return [segment["index"] for segment in sorted_segments]
