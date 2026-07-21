import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from legacy_model_bridge.runtime.nemo_asr import (
    NemoASRRequest,
    NemoASRServiceError,
    NemoASRWarmClient,
    build_nemo_asr_service_command,
    build_nemo_asr_worker_command,
    resolve_nemo_archive,
    transcribe_nemo_asr,
)
from legacy_model_bridge.runtime.workers import load_worker_registry


def _write_nemo_archive(path: Path, target: str = "nemo.collections.asr.models.EncDecRNNTBPEModel") -> None:
    config = path.parent / "model_config.yaml"
    config.write_text(f"target: {target}\n", encoding="utf-8")
    with tarfile.open(path, "w") as archive:
        archive.add(config, arcname="model_config.yaml")


def test_resolve_nemo_archive_from_model_dir(tmp_path: Path) -> None:
    archive = tmp_path / "model.nemo"
    _write_nemo_archive(archive)

    assert resolve_nemo_archive(archive_path=tmp_path) == archive


def test_nemo_asr_dry_run_writes_resolved_archive(tmp_path: Path) -> None:
    archive = tmp_path / "parakeet.nemo"
    _write_nemo_archive(archive)
    request = NemoASRRequest(archive_path=str(archive), output_dir=str(tmp_path / "out"), load_only=True)

    result = transcribe_nemo_asr(request, dry_run=True)

    payload = json.loads(Path(result.request_path).read_text())
    assert result.status == "dry_run"
    assert result.command[:4] == ("conda", "run", "-n", "ai")
    assert payload["archive_path"] == str(archive)
    assert payload["load_only"] is True


def test_worker_can_inspect_archive_without_nemo_import(tmp_path: Path) -> None:
    archive = tmp_path / "parakeet.nemo"
    _write_nemo_archive(archive, target="not_a_real_module.Model")
    script = Path("scripts/nemo_asr_worker.py")
    spec = importlib.util.spec_from_file_location("nemo_asr_worker", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module.inspect_request({"archive_path": str(archive), "restore": False})

    assert payload["nemo_archive"] == str(archive)
    assert payload["nemo_model_target"] == "not_a_real_module.Model"
    assert payload["status"] in {"ready", "needs_nemo_model_specific_env", "env_not_ready"}



def test_nemo_asr_worker_env_can_target_fallback(tmp_path: Path, monkeypatch) -> None:
    request_path = tmp_path / "request.json"
    monkeypatch.setenv("LMB_NEMO_ASR_ENV", "nemo_speech")

    cmd = build_nemo_asr_worker_command(request_path)

    assert cmd[:4] == ("conda", "run", "-n", "nemo_speech")


def test_worker_env_status_accepts_python_311(tmp_path: Path) -> None:
    script = Path("scripts/nemo_asr_worker.py")
    spec = importlib.util.spec_from_file_location("nemo_asr_worker_env_status", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ok, problems = module._env_status({"python": "3.11.11", "torch": "2.10.0", "nemo_toolkit": "2.7.3"})

    assert "python<3.12" not in problems

def test_worker_registry_marks_nemo_as_implemented() -> None:
    worker = load_worker_registry().get("nemo_asr_ai")

    assert worker.status == "verified_warm_transcription_service_ai"
    assert worker.entrypoint == "scripts/nemo_asr_worker.py"
    assert worker.artifact_contract == "warm_transcription_service"


def test_cli_nemo_asr_dry_run(tmp_path: Path) -> None:
    archive = tmp_path / "parakeet.nemo"
    _write_nemo_archive(archive)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "legacy_model_bridge.cli",
            "nemo-asr",
            "run",
            "--archive-path",
            str(archive),
            "--output-dir",
            str(tmp_path / "out"),
            "--load-only",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["env"] == "ai"
    assert payload["command"][8] == "scripts/nemo_asr_worker.py"


def test_build_nemo_service_command_enables_jsonl(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"

    cmd = build_nemo_asr_service_command(request_path)

    assert cmd[:5] == ("conda", "run", "--no-capture-output", "-n", "ai")
    assert cmd[-1] == "--serve-jsonl"


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(
            "\n"
            + "backend noise before json\n"
            + json.dumps({"status": "loaded", "restored_class": "FakeASR"}) + "\n"
            + "\n"
            + json.dumps({"status": "ok", "outputs": [{"audio": "a.wav", "text": "hello"}]}) + "\n"
            + json.dumps({"status": "shutdown"}) + "\n"
        )
        self.stderr = io.StringIO()
        self._returncode = None

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        self._returncode = 0
        return 0

    def terminate(self):
        self._returncode = -15


def test_warm_client_reuses_service_protocol(tmp_path: Path) -> None:
    archive = tmp_path / "parakeet.nemo"
    _write_nemo_archive(archive)
    created = []

    def fake_popen(*args, **kwargs):
        created.append((args, kwargs))
        return _FakeProcess()

    client = NemoASRWarmClient(
        NemoASRRequest(archive_path=str(archive), output_dir=str(tmp_path / "out")),
        popen_factory=fake_popen,
    )

    startup = client.start()
    result = client.transcribe(["a.wav"])
    shutdown = client.close()

    assert startup["status"] == "loaded"
    assert result["outputs"] == [{"audio": "a.wav", "text": "hello"}]
    assert shutdown == {"status": "shutdown"}
    assert created[0][0][0][-1] == "--serve-jsonl"


def test_worker_one_shot_accepts_audio_alias(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "parakeet.nemo"
    audio = tmp_path / "audio.wav"
    _write_nemo_archive(archive)
    audio.write_bytes(b"RIFF")
    script = Path("scripts/nemo_asr_worker.py")
    spec = importlib.util.spec_from_file_location("nemo_asr_worker_alias", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeModel:
        def to(self, device):
            return self

        def cpu(self):
            return self

        def eval(self):
            return None

        def transcribe(self, audio_paths):
            return ["ok text"]

    monkeypatch.setattr(module, "_installed_versions", lambda: {"python": "3.12.0", "torch": "2.13.0", "nemo_toolkit": "2.7.3"})
    monkeypatch.setattr(module, "_env_status", lambda versions: (True, ()))
    monkeypatch.setattr(module, "_target_import_status", lambda target: (True, "ok"))
    monkeypatch.setattr(module, "_restore_model", lambda nemo_path, map_location: FakeModel())

    payload = module.run_request({
        "archive_path": str(archive),
        "audio": [str(audio)],
        "restore_map_location": "cpu",
        "device": "cpu",
    })

    assert payload["status"] == "ok"
    assert payload["transcribe"]["outputs"] == [{"audio": str(audio), "text": "ok text"}]


def test_worker_serve_jsonl_redirects_backend_stdout(tmp_path: Path, monkeypatch, capsys) -> None:
    archive = tmp_path / "parakeet.nemo"
    _write_nemo_archive(archive)
    script = Path("scripts/nemo_asr_worker.py")
    spec = importlib.util.spec_from_file_location("nemo_asr_worker_serve", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeService:
        def __init__(self, req):
            pass

        def restore(self):
            print("backend log line")
            return {"status": "loaded"}

    monkeypatch.setattr(module, "NemoASRWarmService", FakeService)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"action":"shutdown"}\n'))

    assert module.serve_jsonl({"archive_path": str(archive)}) == 0
    captured = capsys.readouterr()
    stdout_lines = [json.loads(line) for line in captured.out.splitlines()]

    assert stdout_lines == [{"status": "loaded"}, {"status": "shutdown"}]
    assert "backend log line" in captured.err


def test_warm_client_times_out_on_silent_pipe(tmp_path: Path) -> None:
    import pytest

    archive = tmp_path / "parakeet.nemo"
    _write_nemo_archive(archive)
    processes = []

    def silent_popen(*args, **kwargs):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append(proc)
        return proc

    client = NemoASRWarmClient(
        NemoASRRequest(archive_path=str(archive), output_dir=str(tmp_path / "out")),
        popen_factory=silent_popen,
    )

    with pytest.raises(NemoASRServiceError, match="timed out"):
        client.start(timeout_sec=0.05)
    client.close(timeout_sec=0.1)
    assert processes[0].poll() is not None


def test_worker_rejects_empty_warm_transcribe_request(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "parakeet.nemo"
    _write_nemo_archive(archive)
    script = Path("scripts/nemo_asr_worker.py")
    spec = importlib.util.spec_from_file_location("nemo_asr_worker_bad_request", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    service = module.NemoASRWarmService({"archive_path": str(archive), "device": "cpu"})

    assert service.transcribe([]) == {"status": "bad_request", "error": "audio_paths must not be empty"}
    assert service.transcribe([1]) == {"status": "bad_request", "error": "audio_paths must be a list of strings"}


def test_worker_status_reports_cached_restore(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "parakeet.nemo"
    _write_nemo_archive(archive)
    script = Path("scripts/nemo_asr_worker.py")
    spec = importlib.util.spec_from_file_location("nemo_asr_worker_cached", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeModel:
        def cpu(self):
            return self

        def eval(self):
            return None

    monkeypatch.setattr(module, "_installed_versions", lambda: {"python": "3.12.0", "torch": "2.13.0", "nemo_toolkit": "2.7.3"})
    monkeypatch.setattr(module, "_env_status", lambda versions: (True, ()))
    monkeypatch.setattr(module, "_target_import_status", lambda target: (True, "ok"))
    monkeypatch.setattr(module, "_restore_model", lambda nemo_path, map_location: FakeModel())

    service = module.NemoASRWarmService({"archive_path": str(archive), "device": "cpu"})
    first = service.restore()
    second = service.restore()

    assert first["status"] == "loaded"
    assert first["restore_cached"] is False
    assert second["restore_cached"] is True
    assert second["service"]["restore_count"] == 1


def test_nemo_prompt_compat_installs_legacy_target_module(monkeypatch) -> None:
    import importlib
    import sys
    import types

    from legacy_model_bridge.runtime import nemo_asr_prompt_compat as compat

    class FakeBase:
        pass

    fake_modules = {
        "torch": types.SimpleNamespace(nn=types.SimpleNamespace(Sequential=object, Linear=lambda *a, **k: object(), ReLU=lambda: object()), transpose=lambda x, a, b: x),
        "omegaconf": types.SimpleNamespace(ListConfig=list, OmegaConf=types.SimpleNamespace(create=lambda x: x), open_dict=lambda cfg: __import__("contextlib").nullcontext(cfg)),
        "nemo.collections.asr.metrics.bleu": types.SimpleNamespace(BLEU=object),
        "nemo.collections.asr.metrics.wer": types.SimpleNamespace(WER=object),
        "nemo.collections.asr.models.rnnt_bpe_models": types.SimpleNamespace(EncDecRNNTBPEModel=FakeBase),
        "nemo.collections.asr.parts.submodules.rnnt_decoding": types.SimpleNamespace(RNNTBPEDecoding=object),
        "nemo.utils": types.SimpleNamespace(logging=types.SimpleNamespace(info=lambda *a, **k: None), model_utils=types.SimpleNamespace(convert_model_config_to_dict_config=lambda cfg: cfg, maybe_update_config_version=lambda cfg: cfg)),
    }

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == compat.SHIM_MODULE:
            raise ModuleNotFoundError(name)
        if name in fake_modules:
            return fake_modules[name]
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, compat.SHIM_MODULE, raising=False)

    assert compat.install_nemo_rnnt_bpe_prompt_shim() is True
    module = importlib.import_module(compat.SHIM_MODULE)
    assert hasattr(module, "EncDecRNNTBPEModelWithPrompt")
    assert module.EncDecRNNTBPEModelWithPrompt.__module__ == compat.SHIM_MODULE


def test_nemo_prompt_compat_skips_unrelated_target(monkeypatch) -> None:
    from legacy_model_bridge.runtime import nemo_asr_prompt_compat as compat

    called = []
    monkeypatch.setattr(compat, "install_nemo_rnnt_bpe_prompt_shim", lambda: called.append(True) or True)

    assert compat.maybe_install_nemo_asr_compat("nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel") == ()
    assert called == []

def test_parakeet_unified_restore_override_translates_old_context_config(tmp_path: Path) -> None:
    archive = tmp_path / "parakeet-unified.nemo"
    config = tmp_path / "model_config.yaml"
    config.write_text(
        "target: nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel\n"
        "encoder:\n"
        "  att_context_size:\n"
        "  - -1\n"
        "  - -1\n"
        "  att_chunk_context_size:\n"
        "  - - 70\n"
        "  - - 1\n"
        "    - 2\n"
        "  - - 0\n"
        "    - 13\n"
        "  att_context_style: chunked_limited_with_rc\n"
        "  conv_context_style: dcc\n",
        encoding="utf-8",
    )
    with tarfile.open(archive, "w") as tar:
        tar.add(config, arcname="model_config.yaml")
    script = Path("scripts/nemo_asr_worker.py")
    spec = importlib.util.spec_from_file_location("nemo_asr_worker_unified_config", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg, patches = module._parakeet_unified_restore_override(
        archive, "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
    )

    assert patches == ("nemo_parakeet_unified_context_config_compat",)
    assert cfg.encoder.att_context_style == "chunked_limited"
    assert list(cfg.encoder.att_context_size) == [-1, 13]
    assert "att_chunk_context_size" not in cfg.encoder
    assert "conv_context_style" not in cfg.encoder

def test_post_restore_compat_defaults_missing_validation_ds() -> None:
    import types

    script = Path("scripts/nemo_asr_worker.py")
    spec = importlib.util.spec_from_file_location("nemo_asr_worker_post_restore", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(validation_ds=None))

    patches = module._apply_post_restore_compat(model)

    assert patches == ("nemo_asr_transcribe_validation_ds_default",)
    assert model.cfg.validation_ds == {}

