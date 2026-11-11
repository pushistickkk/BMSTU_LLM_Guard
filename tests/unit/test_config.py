from pathlib import Path

from llm_guard.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_repo_config_loads():
    cfg = load_config(ROOT / "config" / "guard.yaml")
    assert cfg.backend.model
    assert cfg.backend.base_url.startswith("http")
    assert "ACME" in cfg.app.system_prompt
    assert cfg.guard.enabled is False


def test_inline_prompt_wins(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("app:\n  system_prompt: hello\n  system_prompt_file: missing.txt\n", "utf-8")
    assert load_config(p).app.system_prompt == "hello"


def test_guarded_profile_loads():
    cfg = load_config(ROOT / "config" / "profiles" / "guarded.yaml")
    assert cfg.guard.enabled is True
    assert cfg.server.port == 8001
    assert "ACME" in cfg.app.system_prompt
    assert cfg.guard.input_checks
