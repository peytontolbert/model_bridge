import json
import subprocess
import sys
from pathlib import Path

from legacy_model_bridge.runtime.causal_lm import (
    CausalLMRequest,
    generate_causal_lm,
    read_model_config,
    resolve_model_path,
    should_trust_remote_code,
    should_use_cache,
)
from legacy_model_bridge.runtime.transformers_compat import (
    PATCH_ID_MOBILELLM_LEGACY_CACHE,
    mobilellm_requires_no_cache,
)


def _write_model(path: Path, config: dict) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_resolve_model_path_prefers_model_root(tmp_path: Path) -> None:
    _write_model(tmp_path / "MobileLLM-125M", {"model_type": "mobilellm"})

    assert resolve_model_path("MobileLLM-125M", tmp_path) == tmp_path / "MobileLLM-125M"


def test_mobilellm_policy_disables_cache_and_trusts_remote_code(tmp_path: Path) -> None:
    model = tmp_path / "MobileLLM-125M"
    _write_model(
        model,
        {
            "model_type": "mobilellm",
            "architectures": ["MobileLLMForCausalLM"],
            "auto_map": {"AutoModelForCausalLM": "modeling_mobilellm.MobileLLMForCausalLM"},
            "use_cache": True,
        },
    )
    config = read_model_config(model)

    assert mobilellm_requires_no_cache(config) is True
    assert should_use_cache(config, None) is False
    assert should_trust_remote_code(config, None) is True


def test_causal_lm_dry_run_records_policy_and_patches(tmp_path: Path) -> None:
    _write_model(
        tmp_path / "MobileLLM-125M",
        {
            "model_type": "mobilellm",
            "architectures": ["MobileLLMForCausalLM"],
            "auto_map": {"AutoModelForCausalLM": "modeling_mobilellm.MobileLLMForCausalLM"},
            "use_cache": True,
        },
    )

    result = generate_causal_lm(
        CausalLMRequest(model_id="MobileLLM-125M", model_root=str(tmp_path), dry_run=True)
    )

    assert result.status == "dry_run"
    assert result.trust_remote_code is True
    assert result.use_cache is False
    assert PATCH_ID_MOBILELLM_LEGACY_CACHE in result.applied_patches


def test_cli_causal_lm_dry_run(tmp_path: Path) -> None:
    _write_model(tmp_path / "MobileLLM-125M", {"model_type": "mobilellm", "use_cache": True})

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "legacy_model_bridge.cli",
            "causal-lm",
            "generate",
            "--model-id",
            "MobileLLM-125M",
            "--model-root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "dry_run"
    assert payload["use_cache"] is False


def test_tokenizer_fallback_rejects_bool_auto_tokenizer(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    model = tmp_path / "MobileLLM-125M"
    _write_model(model, {"model_type": "mobilellm"})
    (model / "tokenizer.model").write_text("stub", encoding="utf-8")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return False

    class FakeLlamaTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return lambda text, return_tensors=None: {"input_ids": []}

    fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer, LlamaTokenizer=FakeLlamaTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from legacy_model_bridge.runtime.causal_lm import load_tokenizer_with_fallback
    from legacy_model_bridge.runtime.transformers_compat import PATCH_ID_MOBILELLM_SLOW_TOKENIZER

    tokenizer, patches = load_tokenizer_with_fallback(model, trust_remote_code=True)

    assert callable(tokenizer)
    assert PATCH_ID_MOBILELLM_SLOW_TOKENIZER in patches



def test_mobilellm_family_smoke_skips_missing_model(tmp_path: Path) -> None:
    import argparse

    from scripts.smoke_mobilellm_family import _run_one

    args = argparse.Namespace(
        model_root=str(tmp_path),
        report_dir=str(tmp_path / "reports"),
        max_new_tokens=4,
        env_name="ai",
        cuda_label="cuda0",
        dry_run=True,
    )
    Path(args.report_dir).mkdir()

    result = _run_one(args, "MobileLLM-600M")

    assert result["status"] == "skipped_missing_model_path"
    payload = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert payload["model_id"] == "MobileLLM-600M"


def test_mobilellm_family_smoke_writes_summary_for_dry_run(tmp_path: Path) -> None:
    from scripts.smoke_mobilellm_family import main

    _write_model(tmp_path / "MobileLLM-125M", {"model_type": "mobilellm", "use_cache": True})
    summary = tmp_path / "summary.json"

    assert main([
        "--model", "MobileLLM-125M",
        "--model-root", str(tmp_path),
        "--report-dir", str(tmp_path / "reports"),
        "--summary-json", str(summary),
        "--device", "cpu",
        "--dry-run",
    ]) == 0

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["results"][0]["status"] == "dry_run"



def test_mobilellm_family_default_matrix_contains_targeted_siblings() -> None:
    from scripts.smoke_mobilellm_family import DEFAULT_MODELS

    assert "MobileLLM-125M" in DEFAULT_MODELS
    assert "MobileLLM-125M-layer-share" in DEFAULT_MODELS
    assert "MobileLLM-125MMobileLLM-125M-layer-share" not in DEFAULT_MODELS
    assert "MobileLLM-350M-layer-share" in DEFAULT_MODELS
    assert "MobileLLM-600M" in DEFAULT_MODELS
    assert "MobileLLM-R1.5-950M" in DEFAULT_MODELS
    assert "MobileLLM-Pro-base" in DEFAULT_MODELS
    assert len(DEFAULT_MODELS) == 11


def test_mobilellm_policy_by_config_family() -> None:
    assert should_use_cache({"model_type": "mobilellm", "use_cache": True}, None) is False
    assert should_use_cache({"model_type": "llama", "use_cache": True}, None) is True
    assert should_use_cache({"model_type": "llama4_text", "architectures": ["Llama4ForCausalLM"], "use_cache": True}, None) is True
    assert should_use_cache({"model_type": "llama", "architectures": ["MobileLLMForCausalLM"], "use_cache": True}, None) is False


def test_causal_lm_failure_preserves_resolved_model_path(tmp_path: Path) -> None:
    model = tmp_path / "MobileLLM-125M"
    _write_model(model, {"model_type": "mobilellm"})

    result = generate_causal_lm(
        CausalLMRequest(model_id="MobileLLM-125M", model_root=str(tmp_path), dtype="definitely_invalid")
    )

    assert result.status == "failed"
    assert result.model_path == str(model)
    assert "unsupported dtype" in result.error or result.error


def test_mobilellm_family_contains_paretoq_matrix() -> None:
    from scripts.smoke_mobilellm_family import FAMILY_MODELS, PARETOQ_MODELS

    assert FAMILY_MODELS["paretoq"] == PARETOQ_MODELS
    assert "MobileLLM-ParetoQ-125M-1-bit" in PARETOQ_MODELS
    assert "MobileLLM-ParetoQ-1.5B-BF16" in PARETOQ_MODELS
    assert len(PARETOQ_MODELS) == 18
