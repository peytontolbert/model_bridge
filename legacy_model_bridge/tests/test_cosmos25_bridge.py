from legacy_model_bridge.runtime.cosmos25 import (
    Cosmos25Request,
    build_cosmos25_worker_command,
    default_checkpoint_for_model,
    default_repo_root_for_model,
    family_for_model,
    plan_or_run_cosmos25,
)


def test_cosmos25_resolves_known_models() -> None:
    assert family_for_model("nvidia/Cosmos-Transfer2.5-2B") == "transfer"
    assert family_for_model("nvidia/Cosmos-Predict2.5-14B") == "predict"
    assert str(default_repo_root_for_model("nvidia/Cosmos-Transfer2.5-2B")).endswith("cosmos-transfer2.5")
    assert str(default_checkpoint_for_model("nvidia/Cosmos-Predict2.5-14B")).endswith("_ema_bf16.pt")


def test_cosmos25_worker_command_uses_bridge_env(tmp_path) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"

    cmd = build_cosmos25_worker_command(request_path, result_path)

    assert cmd[:5] == ("conda", "run", "--no-capture-output", "-n", "cosmos25_py310")
    assert "scripts/cosmos25_official_worker.py" in cmd
    assert str(result_path) in cmd


def test_cosmos25_dry_run_writes_request_and_result_path(tmp_path) -> None:
    result = plan_or_run_cosmos25(
        Cosmos25Request(
            model_id="nvidia/Cosmos-Transfer2.5-2B",
            output_dir=str(tmp_path),
            input_files=("assets/robot_example/distilled/edge/robot_edge_spec.json",),
        ),
        dry_run=True,
    )

    assert result.status == "dry_run"
    assert result.dry_run is True
    assert result.request_path.endswith(".json")
    assert result.result_path and result.result_path.endswith(".json")


def test_cosmos25_request_records_gpu_and_offline_policy(tmp_path) -> None:
    result = plan_or_run_cosmos25(
        Cosmos25Request(
            model_id="nvidia/Cosmos-Transfer2.5-2B",
            output_dir=str(tmp_path),
            cuda_visible_devices="1",
            offline_only=True,
        ),
        dry_run=True,
    )

    payload = result.command
    assert "scripts/cosmos25_official_worker.py" in payload


def test_cosmos25_student_only_worker_launches_bridge_wrapper(tmp_path) -> None:
    request = {
        "family": "transfer",
        "repo_root": "/data/clone/third_party/cosmos-transfer2.5",
        "input_files": ("assets/robot_example/distilled/edge/robot_edge_spec.json",),
        "output_dir": str(tmp_path),
        "model": "edge/distilled",
        "checkpoint_path": "/arxiv/models/nvidia/Cosmos-Transfer2.5-2B/distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt",
        "student_only": True,
    }

    from scripts.cosmos25_official_worker import _official_command

    cmd = _official_command(request)

    assert "scripts/cosmos25_transfer_student_only_infer.py" in cmd[1]
    assert "--runtime-root" in cmd
    assert "--disable-guardrails" in cmd
    assert "--offload-diffusion-model" not in cmd
    assert "--offload-text-encoder" not in cmd
    assert "--offload-tokenizer" not in cmd


def test_cosmos25_student_only_rejects_predict(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError, match="student_only"):
        plan_or_run_cosmos25(
            Cosmos25Request(
                model_id="nvidia/Cosmos-Predict2.5-14B",
                output_dir=str(tmp_path),
                student_only=True,
            ),
            dry_run=True,
        )

def test_cosmos25_text_encoder_cpu_offload_borrows_student_net_memory() -> None:
    from legacy_model_bridge.runtime.cosmos25_student_only import _wrap_text_encoder_cpu_offload

    events: list[tuple[str, str]] = []

    class FakeModule:
        def __init__(self, name: str) -> None:
            self.name = name

        def to(self, device: str) -> None:
            events.append((self.name, device))

    class FakeTextEncoder:
        def __init__(self) -> None:
            self.model = FakeModule("text_encoder")
            self.device = "cuda"

        def compute_text_embeddings_online(self, *_args, **_kwargs):
            events.append(("compute", self.device))
            return "embeddings"

    text_encoder = FakeTextEncoder()
    student_net = FakeModule("student_net")

    _wrap_text_encoder_cpu_offload(text_encoder)
    text_encoder._legacy_model_bridge_precompute_cpu_modules = (student_net,)

    assert text_encoder.compute_text_embeddings_online({}, "caption") == "embeddings"

    assert events == [
        ("text_encoder", "cpu"),
        ("student_net", "cpu"),
        ("compute", "cuda"),
        ("text_encoder", "cpu"),
        ("student_net", "cuda"),
    ]
    assert text_encoder.device == "cpu"

def test_cosmos25_tokenizer_cpu_offload_supports_inner_model_to() -> None:
    from legacy_model_bridge.runtime.cosmos25_student_only import _wrap_tokenizer_cpu_offload

    events: list[tuple[str, str]] = []

    class FakeModule:
        def to(self, device: str) -> None:
            events.append(("inner_tokenizer", device))

    class FakeTokenizer:
        def __init__(self) -> None:
            self.model = FakeModule()

    class FakeModel:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def encode(self, state):
            events.append(("encode", state))
            return "latent"

        def decode(self, latent):
            events.append(("decode", latent))
            return "video"

    model = FakeModel()

    _wrap_tokenizer_cpu_offload(model)

    assert model.encode("state") == "latent"
    assert model.decode("latent") == "video"
    assert events == [
        ("inner_tokenizer", "cpu"),
        ("inner_tokenizer", "cuda"),
        ("encode", "state"),
        ("inner_tokenizer", "cpu"),
        ("inner_tokenizer", "cuda"),
        ("decode", "latent"),
        ("inner_tokenizer", "cpu"),
    ]
