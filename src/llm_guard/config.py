from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = "config/guard.yaml"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    api_keys: list[str] = Field(default_factory=list)


class BackendConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3.5:4b"
    timeout_s: float = 180.0
    defaults: dict[str, Any] = Field(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    name: str = "LLM App"
    system_prompt: str = ""
    system_prompt_file: str | None = None


class GuardSettings(BaseModel):
    enabled: bool = False
    device: str = "cuda"
    input_checks: list[str] = Field(default_factory=lambda: ["rules", "hivetrace", "gliner"])
    stop_on_first: bool = True
    parallel: bool = False
    long_input: str = "chunks"  # chunks или tail
    chunk_chars: int = 1500
    chunk_overlap: int = 200
    max_input_chars: int = 30000  # 0 - без ограничения
    secrets: list[str] = Field(default_factory=list)
    output_checks: list[str] = Field(default_factory=lambda: ["secrets", "canary"])
    output_action: str = "block"  # block или log
    similarity_threshold: float = 0.9
    similarity_device: str = "cpu"
    similarity_protect_from: str = ""  # с какой строки промпта сравнивать
    refusal_message: str = ("Извините, с этим я помочь не могу. "
                            "Я могу подсказать по товарам, заказам, доставке и гарантии.")


class LoggingConfig(BaseModel):
    requests_log: str | None = "logs/requests.jsonl"


class Config(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    guard: GuardSettings = Field(default_factory=GuardSettings)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path | None = None):
    cfg_path = Path(path or os.environ.get("LLM_GUARD_CONFIG", DEFAULT_CONFIG_PATH))
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = Config.model_validate(data)

    if cfg.app.system_prompt_file and not cfg.app.system_prompt:
        prompt_path = Path(cfg.app.system_prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = cfg_path.parent / prompt_path
        cfg.app.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    return cfg
