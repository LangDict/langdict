
from langdict import LangDict


def test_langdict_dict():
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
    assert chitchat.as_dict()["messages"] == chitchat_spec["messages"]

    for key in ["llm", "output"]:
        for k, v in chitchat_spec[key].items():
            assert chitchat.as_dict()[key][k] == v
