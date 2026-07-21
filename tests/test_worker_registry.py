import json
import subprocess
import sys

from legacy_model_bridge.runtime.workers import build_worker_preflight_command, load_worker_registry


def test_worker_registry_loads_known_backend_workers() -> None:
    registry = load_worker_registry()

    ids = {worker.worker_id for worker in registry.workers}

    assert "three_d_trellis2_ai_pending_geometry" in ids
    assert "nemo_asr_ai" in ids
    assert "cosmos25_official_py310" in ids
    assert "hunyuan_avatar_ai" in ids


def test_worker_registry_resolves_models_to_workers() -> None:
    registry = load_worker_registry()

    assert registry.for_model("parakeet-rnnt-0.6b").env == "ai"
    assert registry.for_model("nvidia/Cosmos-Predict2.5-14B").expected_python == ("3.10",)
    avatar = registry.for_model("HunyuanVideo-Avatar")
    assert avatar.env == "ai"
    assert avatar.min_cuda_devices == 2
    assert "ffmpeg" in avatar.required_executables
    assert "flash_attn" in avatar.required_imports


def test_worker_preflight_command_includes_required_import_probe() -> None:
    worker = load_worker_registry().get("cosmos25_official_py310")

    cmd = build_worker_preflight_command(worker)

    assert cmd[:4] == ("conda", "run", "-n", "cosmos25_py310")
    assert "flash_attn" in cmd[-1]
    assert "torch_cuda_available" in cmd[-1]
    assert "required_executables" in cmd[-1]
    assert "torch_cuda_device_count" in cmd[-1]
    assert "importlib.metadata.version" in cmd[-1]
    assert "sys.version_info.major" in cmd[-1]


def test_cli_workers_list_includes_non_3d_lanes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "legacy_model_bridge.cli", "workers", "list"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "nemo_asr_ai" in result.stdout
    assert "cosmos25_official_py310" in result.stdout
    assert "hunyuan_avatar_ai" in result.stdout


def test_cli_workers_doctor_can_resolve_model() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "legacy_model_bridge.cli",
            "workers",
            "doctor",
            "HunyuanVideo-Avatar",
            "--model",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["worker_id"] == "hunyuan_avatar_ai"
    assert payload["expected_python"] == ["3.11"]


def test_cosmos_worker_checks_actual_runtime_roots_and_checkpoints() -> None:
    worker = load_worker_registry().get("cosmos25_official_py310")

    assert "/data/clone/third_party/cosmos-predict2.5" in worker.source_paths
    assert "/data/clone/third_party/cosmos-transfer2.5" in worker.source_paths
    assert any(path.endswith("_ema_bf16.pt") for path in worker.required_paths)
    assert "cosmos_predict2.inference" in worker.required_imports
    assert "cosmos_transfer2.inference" in worker.required_imports
    assert worker.status == "implemented"
