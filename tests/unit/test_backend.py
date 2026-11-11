import asyncio
import json

import httpx

from llm_guard.backends.openai_compat import OpenAICompatBackend
from llm_guard.config import BackendConfig


def test_param_priority_and_model():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": []})

    cfg = BackendConfig(base_url="http://llm/v1", model="m1",
                        defaults={"temperature": 0.0, "max_tokens": 10},
                        extra_params={"reasoning_effort": "none"})
    client = httpx.AsyncClient(base_url="http://llm/v1/", transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(cfg, client=client)
    asyncio.run(backend.chat([{"role": "user", "content": "x"}], temperature=0.7, seed=None))

    assert seen["url"] == "http://llm/v1/chat/completions"
    b = seen["body"]
    assert b["model"] == "m1" and b["stream"] is False
    assert b["temperature"] == 0.7
    assert b["max_tokens"] == 10
    assert "seed" not in b
    assert b["reasoning_effort"] == "none"


def test_local_url_detection():
    from llm_guard.backends.openai_compat import is_local_url
    assert is_local_url("http://localhost:11434/v1")
    assert is_local_url("http://127.0.0.1:8000")
    assert not is_local_url("https://api.example.com/v1")
