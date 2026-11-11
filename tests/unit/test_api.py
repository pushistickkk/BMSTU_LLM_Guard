from fastapi.testclient import TestClient

from llm_guard.api.server import create_app
from llm_guard.backends.openai_compat import BackendError
from llm_guard.config import Config


class FakeBackend:

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    async def chat(self, messages, **params):
        self.calls.append((messages, params))
        if self.fail:
            raise BackendError("down")
        return {
            "id": "x", "object": "chat.completion", "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "привет"}}],
        }


def make(api_keys=None, fail=False):
    cfg = Config.model_validate({
        "app": {"system_prompt": "SYS"},
        "server": {"api_keys": api_keys or []},
        "logging": {"requests_log": None},
    })
    backend = FakeBackend(fail=fail)
    return TestClient(create_app(cfg, backend=backend)), backend


def body(text="hi", **kw):
    return {"model": "any", "messages": [{"role": "user", "content": text}], **kw}


def test_health():
    client, _ = make()
    assert client.get("/health").json()["status"] == "ok"


def test_chat_prepends_system_prompt():
    client, backend = make()
    r = client.post("/v1/chat/completions", json=body("вопрос", temperature=0.5))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "привет"
    assert r.json()["guard"]["decision"] == "allow"
    messages, params = backend.calls[0]
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[1] == {"role": "user", "content": "вопрос"}
    assert params["temperature"] == 0.5


def test_stream_rejected():
    client, _ = make()
    assert client.post("/v1/chat/completions", json=body(stream=True)).status_code == 400


def test_empty_messages_rejected():
    client, _ = make()
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422


def test_api_key():
    client, _ = make(api_keys=["secret"])
    assert client.post("/v1/chat/completions", json=body()).status_code == 401
    ok = client.post("/v1/chat/completions", json=body(),
                     headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_backend_error_is_502():
    client, _ = make(fail=True)
    assert client.post("/v1/chat/completions", json=body()).status_code == 502
