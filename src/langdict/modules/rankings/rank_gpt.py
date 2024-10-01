from typing import Any, Dict, List

from langdict import LangDict, LangDictModule


_SPECIFICATION = {
    "messages": [
        ("system", "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."),
        ("human", "I will provide you with {num} passages, each indicated by number identifier []. Rank them based on their relevance to query: {query}."),
        ("ai", "Okay, please provide the passages."),
        ("placeholder", "{passages}"),
        ("human", """Search Query: {query}.
Rank the {num} passages above based on their relevance to the search query.
The passages should be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be List[int], e.g., [1, 2].
Only response the ranking results, do not say any word or explain."""),
    ],
    "llm": {
        "model": "gpt-4o-mini",
    },
    "output": {
        "type": "json"
    },
    "metadata": {
        "arxiv": "https://arxiv.org/abs/2304.09542",
    }
}


class RankGPT(LangDictModule):

    def __init__(self):
        super().__init__(
            LangDict.from_dict(_SPECIFICATION)
        )

    def forward(self, query: str, passages: List[str]) -> Dict[str, Any]:
        passage_prompts = []
        for i, passage in enumerate(passages):
            passage_prompts.append({
                "role": "user",
                "content": f"[{i + 1}] {passage}"
            })
            passage_prompts.append({
                "role": "assistant",
                "content": f"Received passage [{i + 1}]"
            })

        return {
            "query": query,
            "passages": passage_prompts,
            "num": len(passages),
        }
