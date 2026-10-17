from types import SimpleNamespace

from decert.llm import OpenAICompatClient


class StubCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_client(**fields):
    client = OpenAICompatClient(model="m", **fields)
    stub = StubCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return client, stub


def test_no_extra_body_by_default():
    client, stub = make_client()
    assert client.complete([{"role": "user", "content": "hi"}]) == "ok"
    assert "extra_body" not in stub.kwargs


def test_disable_thinking_extra_body():
    client, stub = make_client(disable_thinking=True)
    client.complete([{"role": "user", "content": "hi"}])
    assert stub.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_reasoning_effort_extra_body():
    client, stub = make_client(reasoning_effort="low")
    client.complete([{"role": "user", "content": "hi"}])
    assert stub.kwargs["extra_body"] == {"reasoning_effort": "low"}

