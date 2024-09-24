
<p align="center">
    <img src="https://github.com/LangDict/langdict/blob/main/images/logo.png" style="inline" width=600>
</p>


<h4 align="center">
    LLM Application Framework as a Specification
</h4>

---

# LangDictg

LangDict is a framework for building agents (Compound AI Systems) using only specifications in a Python `dictionary`. The framework is simple and intuitive to use for production.

Building an LLM Application means, in the extreme, adding LLM API calls. In comparison, many frameworks provide complex functionality. LangDict was created with the design philosophy that building LLM applications should be as simple as possible. Build your own LLM Application with minimal understanding of the framework.

<p align="center">
    <img src="https://github.com/LangDict/langdict/blob/main/images/module.png" style="inline" width=800>
</p>

An Agent can be built by connecting multiple Modules. At LangDict, we focus on the intuitive interface, modularity, extensibility, and reusability of [PyTorch](https://github.com/pytorch/pytorch)'s `nn.Module`. If you have experience developing Neural Networks with PyTorch, you will understand how to use it right away.


## Features

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
        self.search = SimpleKeywordSearch(docs=docs)  # Module
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
  <summary>Easy to change trace options (Console, Langfuse)</summary>

```python
# Apply Trace option to all modules
rag = RAG()

# Console Trace
rag.trace(backend="console")

# Langfuse
rag.trace(backend="langfuse")
```

</details>


## Quick Start

Install LangDict:

```python
$ pip install langdict
```

### Example

**Chitchat** (`LangDict`)
- Build LLM Module with the specification.

```python
from langdict import LangDict


chitchat_spec = {
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
chitchat = LangDict.from_dict(chitchat_spec)
chitchat({
    "name": "LangDict",
    "user_input": "What is your name?"
})
>>> 'My name is LangDict. How can I assist you today?'
```

**RAGAgent** (`Module`, `LangDictModule`)
- Build a agent by connecting multiple modules.

```python
from typing import Any, Dict, List

from langdict import Module, LangDictModule


class RAG(Module):

    def __init__(self, docs: List[str]):
        super().__init__()  
        self.query_rewrite = LangDictModule.from_dict(query_rewrite_spec)
        self.search = SimpleRetriever(docs=docs)  # Module
        self.answer = LangDictModule.from_dict(answer_spec)

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
# Trace
rag.trace(backend="langfuse")
```

## Dependencies

LangDict requires the following:

- [`LangChain`](https://github.com/langchain-ai/langchain) - LangDict consists of PromptTemplate + LLM + Output Parser.
    - langchain
    - langchain-core
- [`LiteLLM`](https://github.com/BerriAI/litellm) - Call 100+ LLM APIs in OpenAI format.

### Optional

- [`Langfuse`](https://github.com/langfuse/langfuse) - If you use langfuse with the Trace option, you need to install it separately.
