import json
import subprocess

from scripts import prove_env_removal


def test_prove_env_marks_missing_latest_imports(monkeypatch) -> None:
    monkeypatch.setattr(
        prove_env_removal,
        "_load_workers",
        lambda: [
            {
                "worker_id": "w",
                "env": "legacy",
                "models": ["m"],
                "required_imports": ["missing_pkg"],
                "required_paths": [],
            }
        ],
    )

    def fake_probe(env, imports):
        ok = env == "legacy"
        return {"imports": {"missing_pkg": {"ok": ok}}}

    monkeypatch.setattr(prove_env_removal, "_probe_imports", fake_probe)

    proof = prove_env_removal.prove_env("legacy", latest_env="ai")

    assert proof["safe_to_remove_now"] is False
    assert proof["missing_in_latest_env"] == ["missing_pkg"]


def test_prove_env_without_workers_is_safe_candidate(monkeypatch) -> None:
    monkeypatch.setattr(prove_env_removal, "_load_workers", lambda: [])

    proof = prove_env_removal.prove_env("unused", latest_env="ai")

    assert proof["safe_to_remove_now"] is True
    assert proof["workers"] == []
