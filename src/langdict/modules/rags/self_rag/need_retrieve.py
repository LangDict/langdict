from typing import Any, Dict, Optional

from langdict import LangDict, Module, LangDictModule


_INPUT_ONLY_SPECIFICATION = {
    "messages": [
        ("system", """Given an instruction, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response.
Please answer [yes] or [no] and write an explanation.

## Demonstrations
Instruction: Give three tips for staying healthy.
{{ "need_retrieval": "[yes]", "explanation": "There might be some online sources listing three tips for staying healthy or some reliable sources to explain the effects of different behaviors on health. So retrieving documents is helpful to improve the response to this query." }}

Instruction: Describe a time when you had to make a difficult decision.
{{ "need_retrieval": "[no]", "explanation": "This instruction is asking about some personal experience and thus does not require one to find some external documents." }}

Instruction: Write a short story in third person narration about a protagonist who has to make an important career decision.
{{ "need_retrieval": "[no]", "explanation": "This instruction asks us to write a short story, which does not require external evidence to verify." }}

Instruction: What is the capital of France?
{{ "need_retrieval": "[yes]", "explanation": "While the instruction simply asks us to answer the capital of France, which is a widely known fact, retrieving web documents for this question can still help." }}

Instruction: Find the area of a circle given its radius. Radius = 4
{{ "need_retrieval": "[no]", "explanation": "This is a math question and does not require external evidence." }}

Instruction: Arrange the words in the given sentence to form a grammatically correct sentence. quickly the brown fox jumped
{{ "need_retrieval": "[no]", "explanation": "This task doesn’t require any external evidence, as it is a simple grammatical question." }}

Instruction: Explain the process of cellular respiration in plants.
{{ "need_retrieval": "[yes]", "explanation": "This instruction asks for a detailed description of a scientific concept, and is highly likely that we can find a reliable and useful document to support the response." }}

Instruction: {instruction}
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


_WITH_PRECEDING_SPECIFICATION = {
    "messages": [
        ("system", """You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional).
If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.
Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification.
There are three cases:
- If the output sentence can be verified solely with the evidence, then respond with [continue].
- If the sentence doesn’t require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with [no].
- If additional information is needed to verify the output sentence, respond with [yes].
Please provide explanations for your judgments.

## Demonstrations
- Instruction: Explain the use of word embeddings in Natural Language Processing.
- Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.
- Evidence: Word embedding
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. Methods to generate this mapping include neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear. Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing, sentiment analysis, next token predictions as well and analogy detection.
{{ "rating": "[yes]", "explanation": "The output discusses the applications of word embeddings, while the evidence only discusses the definitions of word embeddings and how they work. Therefore, we need to retrieve other evidence to verify whether the output is correct or not." }}

- Instruction: {instruction}
- Preceding sentences: {preceding}
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



class NeedRetrieve(Module):
    """
    Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

    Decides when to retrieve with R

    Type: [Retrieve]
    Output: {yes, no, continue}
    """

    def __init__(self):
        super().__init__()

        self.input_only = LangDictModule(
            LangDict.from_dict(_INPUT_ONLY_SPECIFICATION)
        )
        self.with_preceding = LangDictModule(
            LangDict.from_dict(_WITH_PRECEDING_SPECIFICATION)
        )

    def forward(
        self,
        instruction: str,
        preceding: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> Dict[str, Any]:

        if (preceding and evidence):
            inputs = {
                "instruction": instruction,
                "preceding": preceding,
                "evidence": evidence
            }
            result = self.with_preceding(inputs)
        else:
            inputs = {
                "instruction": instruction
            }
            result = self.input_only(inputs)

        if
