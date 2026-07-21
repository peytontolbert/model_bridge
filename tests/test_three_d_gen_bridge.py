import json
import subprocess
import sys
from pathlib import Path

from legacy_model_bridge.runtime.three_d_gen import (
    ThreeDGenRequest,
    build_3d_preflight_command,
    build_3d_worker_command,
    compare_trellis_hunyuan3d,
    expected_artifacts,
    generate_3d,
    get_3d_backend,
    list_3d_backends,
)


def test_backends_encode_user_facing_latest_env_and_worker_split() -> None:
    trellis = get_3d_backend("microsoft/TRELLIS.2-4B")
    hunyuan = get_3d_backend("Hunyuan3D-2mv")

    assert trellis.user_env == "ai"
    assert trellis.env == "ai"
    assert trellis.worker_boundary == "process"
    assert hunyuan.user_env == "ai"
    assert hunyuan.env == "ai"
    assert {backend.backend for backend in list_3d_backends()} == {"hunyuan3d", "trellis"}



def test_preflight_command_checks_backend_python() -> None:
    trellis_cmd = build_3d_preflight_command("trellis")
    hunyuan_cmd = build_3d_preflight_command("hunyuan3d")

    assert trellis_cmd[:4] == ("conda", "run", "-n", "ai")
    assert hunyuan_cmd[:4] == ("conda", "run", "-n", "ai")
    assert "sys.version_info.major" in trellis_cmd[-1]

def test_worker_command_routes_trellis_to_ai_env(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"

    cmd = build_3d_worker_command(request, "trellis", result, env_overrides={"CUDA_VISIBLE_DEVICES": "1"})

    assert cmd[:4] == ("conda", "run", "-n", "ai")
    assert "CUDA_VISIBLE_DEVICES=1" in cmd
    assert "PYTHONNOUSERSITE=1" in cmd
    assert "HF_HOME=/data/huggingface" in cmd
    assert "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6" in cmd
    assert cmd[-4:] == ("scripts/three_d_gen_worker.py", "--request-json", str(request), "--result-json") or cmd[-2:] == ("--result-json", str(result))


def test_expected_artifacts_default_to_glb(tmp_path: Path) -> None:
    req = ThreeDGenRequest(
        backend="hunyuan3d",
        image_path="input.png",
        output_dir=str(tmp_path),
        texture=True,
        extra_args={"output_stem": "asset"},
    )

    assert expected_artifacts(req) == {
        "glb": str(tmp_path / "asset.glb"),
        "textured_glb": str(tmp_path / "asset_textured.glb"),
    }


def test_dry_run_writes_request_and_does_not_execute(tmp_path: Path) -> None:
    req = ThreeDGenRequest(
        backend="hunyuan3d",
        image_path="input.png",
        output_dir=str(tmp_path),
        variant="hunyuan3d-dit-v2-mv",
        extra_args={"num_inference_steps": 1},
    )

    result = generate_3d(req, dry_run=True)

    assert result.status == "dry_run"
    assert result.env == "ai"
    assert result.command[:4] == ("conda", "run", "-n", "ai")
    payload = json.loads(Path(result.request_path).read_text())
    assert payload["model_id"] == "Hunyuan3D-2mv"
    assert payload["model_path"] == "/arxiv/models/Hunyuan3D-2mv"
    assert payload["extra_args"] == {"num_inference_steps": 1}



def test_trellis_dry_run_resolves_hf_snapshot_path(tmp_path: Path) -> None:
    req = ThreeDGenRequest(
        backend="trellis",
        image_path="input.png",
        output_dir=str(tmp_path),
        extra_args={"load_only": True},
    )

    result = generate_3d(req, dry_run=True)
    payload = json.loads(Path(result.request_path).read_text())

    assert payload["model_id"] == "microsoft/TRELLIS.2-4B"
    assert "models--microsoft--TRELLIS.2-4B" in payload["model_path"]
    assert "/snapshots/" in payload["model_path"]

def test_conflict_report_marks_trellis_ai_real_image_generation_verified() -> None:
    report = compare_trellis_hunyuan3d()

    assert report.status == "trellis_ai_real_image_tiny_glb_verified"
    assert report.trellis.env == "ai"
    assert report.hunyuan3d.env == "ai"
    assert any("core native stack imports" in conflict or "default flex_gemm backend verified" in conflict for conflict in report.conflicts)


def test_cli_three_d_backends() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "legacy_model_bridge.cli", "three-d", "backends"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "trellis\tmicrosoft/TRELLIS.2-4B" in result.stdout
    assert "hunyuan3d\tHunyuan3D-2mv" in result.stdout


def test_cli_three_d_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "legacy_model_bridge.cli",
            "three-d",
            "run",
            "trellis",
            "--image-path",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--variant",
            "512",
            "--extra-json",
            '{"num_inference_steps": 1}',
            "--cuda-visible-devices",
            "1",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["env"] == "ai"
    assert "CUDA_VISIBLE_DEVICES=1" in payload["command"]
    assert payload["artifacts"]["glb"].endswith("mesh.glb")
