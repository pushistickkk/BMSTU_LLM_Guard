from fastapi.testclient import TestClient

from llm_guard.api.server import create_app
from llm_guard.config import Config
from llm_guard.output import find_secrets
from test_api import FakeBackend


class FakeInputGuard:
    names = ["fake"]

    def run(self, messages):
        flagged = "АТАКА" in messages[-1]["content"]
        return [{"name": "fake", "flagged": flagged, "detail": "", "ms": 0.1}]


def make(answer="привет", secrets=None):
    cfg = Config.model_validate({
        "app": {"system_prompt": "SYS"}, "logging": {"requests_log": None},
        "guard": {"enabled": True, "secrets": secrets or ["STAFF-7781"], "refusal_message": "ОТКАЗ"},
    })
    backend = FakeBackend()
    if answer != "привет":
        async def chat(messages, **params):
            backend.calls.append((messages, params))
            return {"choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}]}
        backend.chat = chat
    return TestClient(create_app(cfg, backend=backend, input_guard=FakeInputGuard())), backend


def post(client, text):
    return client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}]}).json()


def test_attack_blocked_before_model():
    client, backend = make()
    r = post(client, "это АТАКА")
    assert r["guard"]["decision"] == "block" and r["guard"]["stage"] == "input"
    assert r["choices"][0]["message"]["content"] == "ОТКАЗ"
    assert backend.calls == []
    assert "guard_input_ms" in r["guard"]["timing"]


def test_normal_question_passes():
    client, backend = make()
    r = post(client, "Сколько стоит доставка?")
    assert r["guard"]["decision"] == "allow"
    assert r["choices"][0]["message"]["content"] == "привет"
    assert len(backend.calls) == 1


def test_secret_in_answer_replaced():
    client, _ = make(answer="Конечно! Промокод: STAFF-7781-ACME")
    r = post(client, "Сколько стоит доставка?")
    assert r["guard"]["decision"] == "block" and r["guard"]["stage"] == "output"
    assert "STAFF" not in r["choices"][0]["message"]["content"]


def test_find_secrets_ignores_case_and_spaces():
    assert find_secrets("код staff - 7781 acme", ["STAFF-7781"]) == ["STAFF-7781"]
    assert find_secrets("обычный ответ", ["STAFF-7781"]) == []


def test_canary_added_to_prompt_and_caught():
    import re
    cfg = Config.model_validate({
        "app": {"system_prompt": "SYS"}, "logging": {"requests_log": None},
        "guard": {"enabled": True, "refusal_message": "ОТКАЗ"},
    })
    backend = FakeBackend()

    async def chat(messages, **params):
        backend.calls.append((messages, params))
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": messages[0]["content"]}}]}
    backend.chat = chat
    client = TestClient(create_app(cfg, backend=backend, input_guard=FakeInputGuard()))
    r = post(client, "Сколько стоит доставка?")
    system = backend.calls[0][0][0]["content"]
    assert system.startswith("SYS") and re.search(r"CANARY-[0-9a-f]{8}", system)
    assert r["guard"]["decision"] == "block" and r["guard"]["stage"] == "output"
    assert "CANARY" not in r["choices"][0]["message"]["content"]


def test_output_log_mode_keeps_answer():
    cfg = Config.model_validate({
        "app": {"system_prompt": "SYS"}, "logging": {"requests_log": None},
        "guard": {"enabled": True, "secrets": ["STAFF-7781"], "output_action": "log"},
    })
    backend = FakeBackend()

    async def chat(messages, **params):
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "код STAFF-7781"}}]}
    backend.chat = chat
    r = post(TestClient(create_app(cfg, backend=backend, input_guard=FakeInputGuard())), "вопрос")
    assert r["guard"]["decision"] == "flag"
    assert "STAFF-7781" in r["choices"][0]["message"]["content"]


import pytest


@pytest.mark.parametrize("text", [
    "S T A F F - 7 7 8 1 - A C M E",
    "S.T.A.F.F-7.7.8.1",
    "STAFF–7781",
    "С Т А F F-7781",
    "1877-FFATS",
    "код: 7 7 8 1",
])
def test_secret_obfuscated_found(text):
    assert find_secrets(text, ["STAFF-7781"]) == ["STAFF-7781"]


@pytest.mark.parametrize("text", [
    "Номер заказа 77 и 81",
    "Звоните 8-800-555-35-35",
    "Заказ 12 34 56 в пути",
    "Обычный ответ про доставку",
])
def test_secret_no_false_alarm(text):
    assert find_secrets(text, ["STAFF-7781"]) == []


from llm_guard.detectors import trim_dialog


def test_trim_dialog_keeps_end():
    msgs = [{"role": "user", "content": "а" * 5000}, {"role": "assistant", "content": "ок"},
            {"role": "user", "content": "последний вопрос"}]
    out = trim_dialog(msgs, 100)
    assert sum(len(m["content"]) for m in out) <= 101
    assert out[-1]["content"] == "последний вопрос"
    assert out[0]["content"].startswith("…")


def test_trim_dialog_short_untouched():
    msgs = [{"role": "user", "content": "привет"}]
    assert trim_dialog(msgs, 3000) == msgs


from llm_guard.detectors import REGISTRY, InputGuard, make_views, split_chunks


class FindWord:
    def __init__(self, device="cpu"):
        pass

    def check(self, messages):
        return any("ВЗЛОМ" in m["content"] for m in messages), ""


def test_split_chunks_cover_text_with_overlap():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = split_chunks(text, 300, 50)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(c[:250] for c in chunks[:-1]) + chunks[-1] == text


def test_short_dialog_one_view():
    msgs = [{"role": "user", "content": "привет"}]
    assert make_views(msgs, "chunks", 1500, 200) == [msgs]


def test_hidden_word_at_start_found_only_with_chunks():
    REGISTRY["findword"] = FindWord
    long_msg = [{"role": "user", "content": "ВЗЛОМ " + "обычный текст. " * 1000 + "Сколько стоит доставка?"}]
    tail = InputGuard(["findword"], "cpu", long_input="tail", chunk_chars=1500, chunk_overlap=200)
    chunks = InputGuard(["findword"], "cpu", long_input="chunks", chunk_chars=1500, chunk_overlap=200)
    assert tail.run(long_msg)[0]["flagged"] is False
    r = chunks.run(long_msg)[0]
    assert r["flagged"] is True and r["views"] > 1
