
<p align="center">
    <img src="https://github.com/LangDict/langdict/blob/main/images/logo.png" style="inline" width=600>
</p>


<h4 align="center">
    Build complex LLM Applications with Python Dictionary
</h4>

---

# LangDict

LangDict is a framework for building agents (Compound AI Systems) using only specifications in a Python `Dictionary`. The framework is simple and intuitive to use for production.

The prompts are similar to a feature specification, which is all you need to build an LLM Module. LangDict was created with the design philosophy that building LLM applications should be as simple as possible. Build your own LLM Application with minimal understanding of the framework.

<p align="center">
    <img src="https://github.com/LangDict/langdict/blob/main/images/module.png" style="inline" width=800>
</p>

An Agent can be built by connecting multiple Modules. At LangDict, we focus on the intuitive interface, modularity, extensibility, and reusability of [PyTorch](https://github.com/pytorch/pytorch)'s `nn.Module`. If you have experience developing Neural Networks with PyTorch, you will understand how to use it right away.


## Modules

| Task | Name | Code |
| ---- | ---- | ---- |
| `Ranking` | RankGPT | [Code](https://github.com/LangDict/langdict/blob/main/src/langdict/modules/rankings/rank_gpt.py) |
| `Compression` | TextCompressor (LLMLingua-2) | [Code](https://github.com/LangDict/langdict/blob/main/src/langdict/modules/compressions/llm_lingua2.py) |
| `RAG` | SELF-RAG | [Code](https://github.com/LangDict/langdict/blob/main/src/langdict/modules/rags/self_rag/__init__.py) |


## Key Features

<details>
  <summary>LLM Applicaiton framework for simple, intuitive, specification-based development</summary>

```python
chitchat = LangDict.from_dict({
    "messages": [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ],
    "llm": {
        "model": "gpt-4o-mini",
        "max_tokens": 200
    },
    "output": {
        "type": "string"
    }
})
# format placeholder is key of input dictionary
chitchat({
    "name": "LangDict",
    "user_input": "What is your name?"
})
```

</details>

<details>
  <summary>Simple interface (Stream / Batch) </summary>

```python
rag = RAG()

single_inputs = {
    "conversation": [{"role": "user", "content": "How old is Obama?"}]
}
# invoke
rag(single_inputs)

# stream
rag(single_inputs, stream=True)

# batch
batch_inputs = [{ ...  }, { ...}, ...]
rag(batch_inputs, batch=True)
```

</details>

<details>
  <summary>Modularity: Extensibility, Modifiability, Reusability</summary>

```python
class RAG(Module):

    def __init__(self, docs: List[str]):
        super().__init__()
        self.query_rewrite = LangDictModule.from_dict({ ... })  # Module
        self.search = Retriever(docs=docs)  # Module
        self.answer = LangDictModule.from_dict({ ... })  # Module

    def forward(self, inputs: Dict):
        query_rewrite_result = self.query_rewrite({
            "conversation": inputs["conversation"],
        })
        doc = self.search(query_rewrite_result)
        return self.answer({
            "conversation": inputs["conversation"],
            "context": doc,
        })
```

</details>

<details>
  <summary>Easy to change trace options (Console, Langfuse, LangSmith)</summary>

```python
# Apply Trace option to all modules
rag = RAG()

# Console Trace
rag.trace(backend="console")

# Langfuse
rag.trace(backend="langfuse")

# LangSmith
rag.trace(backend="langsmith")
```

</details>

<details>
  <summary>Easy to change hyper-paramters (Prompt, Paramter)</summary>

```python
rag = RAG()
rag.save_json("rag.json")
# Modify "rag.json" file
rag.load_json("rag.json")
```
</details>

## Quick Start

Install LangDict:

```python
$ pip install langdict
```

### Example

**LangDict**
- Build LLM Module with the specification.

```python
from langdict import LangDict


_SPECIFICATION = {
    "messages": [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ],
    "llm": {
        "model": "gpt-4o-mini",
        "max_tokens": 200
    },
    "output": {
        "type": "string"
    }
}
chitchat = LangDict.from_dict(_SPECIFICATION)
chitchat({
    "name": "LangDict",
    "user_input": "What is your name?"
})
>>> 'My name is LangDict. How can I assist you today?'
```

**Module**
- Build a agent by connecting multiple modules.

```python
from typing import Any, Dict, List

from langdict import Module, LangDictModule


_QUERY_REWRITE_SPECIFICATION = { ... }
_ANSWER_SPECIFICATIOn = { ... }


class RAG(Module):

    def __init__(self, docs: List[str]):
        super().__init__()  
        self.query_rewrite = LangDictModule.from_dict(_QUERY_REWRITE_SPECIFICATION)
        self.search = SimpleRetriever(docs=docs)  # Module
        self.answer = LangDictModule.from_dict(_ANSWER_SPECIFICATIOn)

    def forward(self, inputs: Dict[str, Any]):
        query_rewrite_result = self.query_rewrite({
            "conversation": inputs["conversation"],
        })
        doc = self.search(query_rewrite_result)
        return self.answer({
            "conversation": inputs["conversation"],
            "context": doc,
        })

rag = RAG()
inputs = {
    "conversation": [{"role": "user", "content": "How old is Obama?"}]
}

rag(inputs)
>>> 'Barack Obama was born on August 4, 1961. As of now, in September 2024, he is 63 years old.'
```

- Streaming

```python
rag = RAG()
# Stream
for token in rag(inputs, stream=True):
    print(f"token > {token}")
>>>
token > Bar
token > ack
token >  Obama
token >  was
token >  born
token >  on
token >  August
token >  
token > 4
...
```

- Get observability with a single line of code.

```python
rag = RAG()
# Trace
rag.trace(backend="console")
```

- Save and load the module as a JSON file.

```python
rag = RAG()
rag.save_json("rag.json")
rag.load_json("rag.json")
```

## Dependencies

LangDict requires the following:

- [`LangChain`](https://github.com/langchain-ai/langchain) - LangDict consists of PromptTemplate + LLM + Output Parser.
    - langchain
    - langchain-core
- [`LiteLLM`](https://github.com/BerriAI/litellm) - Call 100+ LLM APIs in OpenAI format.

### Optional

- [`Langfuse`](https://github.com/langfuse/langfuse) - If you use langfuse with the Trace option, you need to install it separately.
