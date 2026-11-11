"""Клиент к серверу с OpenAI-совместимым API (Ollama, vLLM)"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from llm_guard.config import BackendConfig


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local_url(url):
    return (urlparse(url).hostname or "") in LOCAL_HOSTS


class BackendError(Exception):
    pass


class OpenAICompatBackend:
    def __init__(self, cfg, client=None):
        self.cfg = cfg
        self._client = client or httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/") + "/",
            timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            trust_env=not is_local_url(cfg.base_url),
        )

    @property
    def model(self):
        return self.cfg.model

    async def chat(self, messages, **params):
        body: dict[str, Any] = dict(self.cfg.defaults)
        body.update({k: v for k, v in params.items() if v is not None})
        body.update(self.cfg.extra_params)
        body["model"] = self.cfg.model
        body["messages"] = messages
        body["stream"] = False

        try:
            resp = await self._client.post("chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise BackendError(f"нет связи с LLM-сервером {self.cfg.base_url}: {exc!r}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"LLM-сервер вернул {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    async def aclose(self):
        await self._client.aclose()
