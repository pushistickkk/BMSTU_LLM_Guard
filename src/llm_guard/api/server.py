from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from llm_guard import __version__
from llm_guard.api.schemas import ChatCompletionRequest, ChatMessage
from llm_guard.backends.openai_compat import BackendError, OpenAICompatBackend
from llm_guard.config import Config, load_config
from llm_guard.core.request_log import RequestLogger
from llm_guard.output import OutputGuard


def build_messages(system_prompt, messages):
    result: list[dict[str, Any]] = []

    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    result.extend(m.model_dump(exclude_none=True) for m in messages)

    return result


def _last_user_text(messages):
    for m in reversed(messages):
        if m.role == "user":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def refusal_response(model, text, request_id):
    return {
        "id": f"guard-{request_id}", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def create_app(config=None, backend=None, input_guard=None, output_guard=None):
    config = config or load_config()
    backend = backend or OpenAICompatBackend(config.backend)
    req_log = RequestLogger(config.logging.requests_log)
    guard_cfg = config.guard

    if guard_cfg.enabled and input_guard is None and guard_cfg.input_checks:
        from llm_guard.detectors import InputGuard
        print(f"Загрузка проверок: {guard_cfg.input_checks} на {guard_cfg.device}")
        input_guard = InputGuard(guard_cfg.input_checks, guard_cfg.device,
                                 guard_cfg.stop_on_first, guard_cfg.parallel, guard_cfg.long_input,
                                 guard_cfg.chunk_chars, guard_cfg.chunk_overlap)

    if guard_cfg.enabled and output_guard is None and guard_cfg.output_checks:
        output_guard = OutputGuard(guard_cfg, config.app.system_prompt, input_guard)
    # канарейка только в запросе к модели, /health считает хеш без неё
    system_prompt = config.app.system_prompt + (output_guard.system_prompt_suffix() if output_guard else "")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if hasattr(backend, "aclose"):
            await backend.aclose()

    app = FastAPI(title="LLM Guard Proxy", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.backend = backend

    def check_api_key(authorization: str | None = Header(default=None)):
        keys = config.server.api_keys
        if not keys:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token not in keys:
            raise HTTPException(status_code=401, detail="неверный API-ключ")

    @app.get("/health")
    async def health():
        prompt_sha = hashlib.sha256(config.app.system_prompt.encode("utf-8")).hexdigest()[:12]
        return {"status": "ok", "model": config.backend.model, "guard": guard_cfg.enabled,
                "checks": input_guard.names if (guard_cfg.enabled and input_guard) else [],
                "output_checks": output_guard.names if (guard_cfg.enabled and output_guard) else [],
                "system_prompt_sha": prompt_sha}

    @app.get("/v1/models", dependencies=[Depends(check_api_key)])
    async def models():
        return {
            "object": "list",
            "data": [{"id": config.backend.model, "object": "model", "owned_by": "llm-guard"}],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
    async def chat_completions(req: ChatCompletionRequest):
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true пока не поддерживается")

        t_start = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        timing = {}
        checks = []

        client_msgs = [m.model_dump(exclude_none=True) for m in req.messages]
        total_chars = sum(len(str(m.get("content") or "")) for m in client_msgs)
        if guard_cfg.enabled and guard_cfg.max_input_chars and total_chars > guard_cfg.max_input_chars:
            req_log.write({"id": request_id, "decision": "block", "stage": "length", "chars": total_chars})
            raise HTTPException(status_code=413, detail=f"Сообщение слишком длинное: {total_chars} символов, "
                                                        f"максимум {guard_cfg.max_input_chars}")
        # проверка входа
        if guard_cfg.enabled and input_guard is not None:
            t0 = time.perf_counter()
            checks = await asyncio.to_thread(input_guard.run, client_msgs)
            timing["guard_input_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if any(c["flagged"] for c in checks):
                response = refusal_response(config.backend.model, guard_cfg.refusal_message, request_id)
                timing["server_total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
                response["guard"] = {"enabled": True, "decision": "block", "stage": "input",
                                     "checks": checks, "timing": timing}
                req_log.write({"id": request_id, "model": config.backend.model, "decision": "block",
                               "stage": "input", "user": _last_user_text(req.messages)[:1000],
                               "checks": checks})
                return response

        messages = build_messages(system_prompt, req.messages)
        params = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "seed": req.seed,
        }
        t0 = time.perf_counter()
        try:
            response = await backend.chat(messages, **params)
        except BackendError as exc:
            req_log.write({"id": request_id, "error": str(exc)})
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        latency_ms = round((time.perf_counter() - t0) * 1000)

        timing["backend_ms"] = latency_ms
        answer = ""
        try:
            answer = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            pass

        # проверка ответа
        decision, stage = "allow", None
        if guard_cfg.enabled and output_guard is not None:
            t0 = time.perf_counter()
            out_checks = await asyncio.to_thread(output_guard.run, client_msgs, answer)
            timing["guard_output_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            checks.extend(out_checks)
            if any(c["flagged"] for c in out_checks):
                stage = "output"
                if guard_cfg.output_action == "block":
                    decision = "block"
                    response["choices"][0]["message"]["content"] = guard_cfg.refusal_message
                else:
                    decision = "flag"
        response["guard"] = {"enabled": guard_cfg.enabled, "decision": decision, "stage": stage,
                             "checks": checks}
        req_log.write({
            "id": request_id,
            "model": config.backend.model,
            "latency_ms": latency_ms,
            "user": _last_user_text(req.messages)[:1000],
            "answer": answer[:1000],
            "decision": decision,
            "stage": stage,
            "checks": checks,
        })
        timing["server_total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        response["guard"]["timing"] = timing
        return response

    return app
