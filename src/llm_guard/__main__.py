from __future__ import annotations

import argparse

import uvicorn

from llm_guard.api.server import create_app
from llm_guard.config import load_config


def main():
    parser = argparse.ArgumentParser(description="LLM Guard Proxy")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу")
    parser.add_argument("--model", default=None, help="модель вместо backend.model")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg.backend.model = args.model
    print(f"Модель: {cfg.backend.model} @ {cfg.backend.base_url}; защита: {cfg.guard.enabled}")
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
