from typing import Any, Dict

from langdict import LangDict, LangDictModule


_SPECIFICATION = {
    "messages": [
        ("system", """You will receive an instruction, evidence, and output, and optional preceding sentences.
If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.
Your task is to evaluate if the output is fully supported by the information provided in the evidence.

Use the following entailment scale to generate a score:
- [Fully supported] - All information in output is supported by the evidence, or extractions
from the evidence. This is only applicable when the output and part of the evidence are
almost identical.
- [Partially supported] - The output is supported by the evidence to some extent, but there
is major information in the output that is not discussed in the evidence. For example, if an
instruction asks about two concepts and the evidence only discusses either of them, it should
be considered a [Partially supported].
- [No support / Contradictory] - The output completely ignores evidence, is unrelated to the
evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the
instruction.

Make sure to not use any external information/knowledge to judge whether the output is true or not.
Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.

## Demonstrations
- Instruction: Explain the use of word embeddings in Natural Language Processing.
- Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.
- Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.
- Evidence: Word embedding
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. Methods to generate this mapping include neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear. Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing, sentiment analysis, next token predictions as well and analogy detection.
{{ "rating": "[Fully supported]", "explanation": "The output sentence discusses the application of word embeddings, and the evidence mentions all of the applications syntactic parsing, sentiment analysis, next token predictions as well as analogy detection as the applications. Therefore, the score should be [Fully supported]." }}

- Instruction: {instruction}
- Preceding sentences: {preceding}
- Output: {output}
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


class IsSupport(LangDictModule):
    """
    Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

    All of the verification-worthy statement in y is supported by d.

    Type: [IsSup]
    Output: {fully supported, partially supported, no support}
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
