
from langdict import LangDict
from langdict.builders import ChatPromptMessagesBuilder


def test_langdict_openai_chitchat():
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
    print(
        chitchat({
            "name": "LangDict",
            "user_input": "What is your name?"
        })
    )


def test_langdict_openai_query_rewrite():
    query_rewrite = LangDict.from_dict({
        "messages": ChatPromptMessagesBuilder.build(
            persona="You are a helpful AI bot.",
            task_instruction="Rewrite Human's question to search query.",
            output_format="json, query: str",
            fewshot_examples=[],
            conversation_key="conversation"
        ),
        "llm": {
            "model": "gpt-4o-mini",
            "max_tokens": 200
        },
        "output": {
            "type": "json"
        }
    })

    result = query_rewrite({
        "conversation": [{"role": "user", "content": "How old is Obama?"}]
    })

    assert ("query" in result) is True
