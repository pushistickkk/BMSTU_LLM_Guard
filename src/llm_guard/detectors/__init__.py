from __future__ import annotations

import re
import threading
import time
import unicodedata


def dialog_text(messages):
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


class Rules:
    whole_text = True

    INVISIBLE = re.compile("[​-‏⁠-⁤﻿]")
    PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
        r"ignore (all |the )?(previous|above|prior) (instructions|rules)",
        r"(забудь|игнорируй) (все |свои )?(предыдущие |прошлые )?(инструкции|правила)",
        r"(system|системн\w*) (prompt|промпт|инструкц\w*)",
        r"you are now|developer mode|\bDAN\b",
        r"[A-Za-z0-9+/]{40,}={0,2}",  # base64
    ]]

    def __init__(self, device: str = "cpu"):
        pass

    def check(self, messages):
        t = unicodedata.normalize("NFKC", self.INVISIBLE.sub("", dialog_text(messages)))
        hits = [p.pattern[:25] for p in self.PATTERNS if p.search(t)]
        return bool(hits), "; ".join(hits)


class PromptGuard:
    MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

    def __init__(self, device: str = "cuda", threshold: float = 0.5):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device, self.threshold = torch, device, threshold
        self.tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.tok.truncation_side = "left"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID).to(device).eval()

    def check(self, messages):
        with self.torch.inference_mode():
            x = self.tok(dialog_text(messages), return_tensors="pt", truncation=True,
                         max_length=512).to(self.device)
            p = self.model(**x).logits.softmax(-1)[0, -1].item()
        return p >= self.threshold, f"{p:.3f}"


class HiveTrace:
    MODEL_ID = "hivetrace/HiveTraceGuard-Pro"

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID, torch_dtype=dtype).to(device).eval()

    def check(self, messages):
        with self.torch.inference_mode():
            prompt = self.tok.apply_chat_template(messages, tokenize=False)
            x = self.tok(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**x, max_new_tokens=1, do_sample=False)
        ans = self.tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        return ans == "unsafe", ans or "?"


class Gliner:
    MODEL_ID = "hivetrace/gliner-guard-omni"
    ADVERSARIAL = ["none", "prompt_injection", "instruction_override", "jailbreak_persona",
                   "jailbreak_roleplay", "jailbreak_hypothetical", "system_prompt_extraction",
                   "data_exfiltration", "context_manipulation", "token_manipulation",
                   "social_engineering", "multi_turn_escalation"]
    HARMFUL = ["none", "political", "drugs", "weapons", "violence", "cybersecurity",
               "dangerous_instructions", "fraud"]

    def __init__(self, device: str = "cuda"):
        import torch
        from gliner2 import GLiNER2
        self.torch = torch
        self.model = GLiNER2.from_pretrained(self.MODEL_ID)
        self.model.to(device).eval()
        self.schema = (self.model.create_schema()
                       .classification(task="safety", labels=["safe", "unsafe"])
                       .classification(task="adversarial", labels=self.ADVERSARIAL, multi_label=True)
                       .classification(task="harmful", labels=self.HARMFUL, multi_label=True))

    def check(self, messages):
        with self.torch.inference_mode():
            r = self.model.extract(text=dialog_text(messages), schema=self.schema)

        def keep(v):
            v = v if isinstance(v, list) else [v]
            return [x for x in v if x and x != "none"]

        adv, harm = keep(r.get("adversarial", [])), keep(r.get("harmful", []))
        flagged = r.get("safety") == "unsafe" or bool(adv) or bool(harm)
        return flagged, f"{r.get('safety')} | {','.join(adv) or '-'} | {','.join(harm) or '-'}"


REGISTRY = {"rules": Rules, "prompt_guard": PromptGuard, "hivetrace": HiveTrace, "gliner": Gliner}


def trim_dialog(messages: list[dict], max_chars: int | None):
    # конец диалога не длиннее max_chars
    if not max_chars:
        return messages
    out, used = [], 0
    for m in reversed(messages):
        text = m["content"]
        if used + len(text) <= max_chars:
            out.append(m)
            used += len(text)
        else:
            rest = max_chars - used
            if rest > 0 or not out:
                out.append({"role": m["role"], "content": "…" + text[-max(rest, 1):]})
            break
    return list(reversed(out))


def split_chunks(text: str, size: int, overlap: int):
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def make_views(messages: list[dict], mode: str, chunk_chars: int, overlap: int):
    # tail - только конец, chunks - конец + весь текст кусками
    if sum(len(m["content"]) for m in messages) <= chunk_chars:
        return [messages]
    views = [trim_dialog(messages, chunk_chars)]
    if mode == "chunks":
        views += [[{"role": "user", "content": c}]
                  for c in split_chunks(dialog_text(messages), chunk_chars, overlap)]
    return views


def clean_messages(messages: list[dict]):
    out = []
    for m in messages:
        c = m.get("content")
        out.append({"role": m.get("role", "user"), "content": c if isinstance(c, str) else str(c or "")})
    return out


class InputGuard:
    def __init__(self, names: list[str], device: str = "cuda", stop_on_first: bool = True,
                 parallel: bool = False, long_input: str = "chunks", chunk_chars: int = 1500,
                 chunk_overlap: int = 200):
        import contextlib
        from concurrent.futures import ThreadPoolExecutor

        torch = None
        try:
            import torch
        except ImportError:
            pass
        if device == "cuda" and (torch is None or not torch.cuda.is_available()):
            print("Внимание: видеокарта не видна, проверки работают на процессоре")
            device = "cpu"
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise ValueError(f"Неизвестные проверки: {unknown}. Доступны: {list(REGISTRY)}")
        self.device, self.stop_on_first, self.parallel = device, stop_on_first, parallel
        self.long_input, self.chunk_chars, self.chunk_overlap = long_input, chunk_chars, chunk_overlap
        self.checks = []
        for n in names:
            t0 = time.perf_counter()
            self.checks.append((n, REGISTRY[n](device="cpu" if n == "rules" else device)))
            print(f"  проверка {n} загружена за {time.perf_counter() - t0:.1f} с")
        self._locks = {n: threading.Lock() for n in names}
        if device == "cuda":
            self._streams = {n: torch.cuda.Stream() for n in names}
            self._stream_ctx = lambda n: torch.cuda.stream(self._streams[n])
        else:
            self._stream_ctx = lambda n: contextlib.nullcontext()
        self._pool = ThreadPoolExecutor(max_workers=4 * len(names)) if parallel else None
        print(f"  режим: {'параллельный' if parallel else 'последовательный'}")

    @property
    def names(self):
        return [n for n, _ in self.checks]

    def _views(self, chk, msgs):
        if getattr(chk, "whole_text", False):
            return [msgs]
        return make_views(msgs, self.long_input, self.chunk_chars, self.chunk_overlap)

    def _run_one(self, name, chk, msgs):
        views = self._views(chk, msgs)
        t0 = time.perf_counter()
        with self._locks[name]:
            t_start = time.perf_counter()
            flagged, detail = False, ""
            try:
                with self._stream_ctx(name):
                    for i, view in enumerate(views):
                        flagged, detail = chk.check(view)
                        if flagged:
                            if len(views) > 1:
                                detail = f"{detail} (кусок {i + 1} из {len(views)})"
                            break
            except Exception as exc:
                flagged, detail = False, f"ошибка: {type(exc).__name__}: {exc}"[:200]
        t_end = time.perf_counter()
        return {"name": name, "flagged": bool(flagged), "detail": detail, "views": len(views),
                "ms": round((t_end - t_start) * 1000, 1),
                "wait_ms": round((t_start - t0) * 1000, 1)}

    def has(self, name: str):
        return name in self._locks

    def run_one(self, name: str, messages: list[dict]):
        chk = dict(self.checks)[name]
        return self._run_one(name, chk, clean_messages(messages))

    def run(self, messages: list[dict]):
        msgs = clean_messages(messages)
        if self.parallel:
            futures = [self._pool.submit(self._run_one, n, c, msgs) for n, c in self.checks]
            return [f.result() for f in futures]
        results = []
        for name, chk in self.checks:
            r = self._run_one(name, chk, msgs)
            results.append(r)
            if r["flagged"] and self.stop_on_first:
                break
        return results
