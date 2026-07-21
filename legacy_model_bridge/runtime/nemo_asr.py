from __future__ import annotations

import io
import json
import os
import selectors
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TextIO
from uuid import uuid4

from legacy_model_bridge.registry import REPO_ROOT


DEFAULT_MODEL_ROOT = Path("/arxiv/models")
DEFAULT_WORKER_ENTRYPOINT = Path("scripts/nemo_asr_worker.py")
DEFAULT_NEMO_ENV = "ai"


def nemo_asr_env() -> str:
    return os.environ.get("LMB_NEMO_ASR_ENV", DEFAULT_NEMO_ENV)


@dataclass(frozen=True)
class NemoASRRequest:
    model_id: str | None = None
    archive_path: str | None = None
    audio_paths: tuple[str, ...] = ()
    output_dir: str = "/data/tmp/legacy_model_bridge_nemo_asr"
    map_location: str = "cpu"
    device: str = "cuda:0"
    restore: bool = True
    load_only: bool = False
    force_restore: bool = False
    extra_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class NemoASRResult:
    status: str
    env: str
    command: tuple[str, ...]
    request_path: str
    output_dir: str
    result_path: str | None = None
    returncode: int | None = None
    elapsed_sec: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None
    payload: dict[str, Any] | None = None
    dry_run: bool = False


class NemoASRServiceError(RuntimeError):
    pass


def resolve_nemo_archive(model_id: str | None = None, archive_path: str | Path | None = None, model_root: str | Path = DEFAULT_MODEL_ROOT) -> Path | None:
    if archive_path:
        candidate = Path(archive_path)
        if candidate.is_file() and candidate.suffix == ".nemo":
            return candidate
        if candidate.is_dir():
            matches = sorted(candidate.glob("*.nemo")) or sorted(candidate.glob("**/*.nemo"))
            return matches[0] if matches else None
        return candidate
    if not model_id:
        return None
    roots = [Path(model_root) / model_id]
    if "/" in model_id:
        roots.append(Path(model_root) / model_id.split("/", 1)[1])
    for root in roots:
        if root.is_file() and root.suffix == ".nemo":
            return root
        if root.is_dir():
            matches = sorted(root.glob("*.nemo")) or sorted(root.glob("**/*.nemo"))
            if matches:
                return matches[0]
    return None


def _request_payload(request: NemoASRRequest) -> dict[str, Any]:
    payload = asdict(request)
    archive = resolve_nemo_archive(request.model_id, request.archive_path)
    payload["archive_path"] = str(archive) if archive else request.archive_path
    payload["audio"] = list(request.audio_paths)
    payload["restore_map_location"] = request.map_location
    if request.model_id is None and archive:
        payload["model_id"] = archive.stem
    return payload


def write_nemo_asr_request(request: NemoASRRequest, output_dir: str | Path | None = None) -> Path:
    out_dir = Path(output_dir or request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    request_path = out_dir / f"nemo_asr_request_{uuid4().hex}.json"
    request_path.write_text(json.dumps(_request_payload(request), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return request_path


def build_nemo_asr_worker_command(
    request_path: str | Path,
    result_path: str | Path | None = None,
    *,
    serve_jsonl: bool = False,
) -> tuple[str, ...]:
    cmd = [
        "conda",
        "run",
    ]
    if serve_jsonl:
        cmd.append("--no-capture-output")
    cmd.extend([
        "-n",
        nemo_asr_env(),
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONPATH=.",
        "python",
        str(DEFAULT_WORKER_ENTRYPOINT),
        "--request-json",
        str(request_path),
    ])
    if result_path is not None:
        cmd.extend(["--result-json", str(result_path)])
    if serve_jsonl:
        cmd.append("--serve-jsonl")
    return tuple(cmd)


def build_nemo_asr_service_command(request_path: str | Path) -> tuple[str, ...]:
    return build_nemo_asr_worker_command(request_path, serve_jsonl=True)


def transcribe_nemo_asr(request: NemoASRRequest, *, dry_run: bool = False, timeout_sec: int | None = None) -> NemoASRResult:
    out_dir = Path(request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = write_nemo_asr_request(request, out_dir)
    result_path = out_dir / f"nemo_asr_result_{req_path.stem.rsplit('_', 1)[-1]}.json"
    cmd = build_nemo_asr_worker_command(req_path, result_path)
    if dry_run:
        return NemoASRResult(
            status="dry_run",
            env=nemo_asr_env(),
            command=cmd,
            request_path=str(req_path),
            output_dir=str(out_dir),
            result_path=str(result_path),
            dry_run=True,
        )
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout_sec, check=False)
    elapsed = time.monotonic() - start
    status = "ok" if proc.returncode == 0 else "failed"
    error = None if proc.returncode == 0 else f"worker exited {proc.returncode}"
    payload: dict[str, Any] | None = None
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            status = str(payload.get("status", status))
            error = payload.get("error", error)
        except json.JSONDecodeError:
            error = error or "worker result JSON was invalid"
    return NemoASRResult(
        status=status,
        env=nemo_asr_env(),
        command=cmd,
        request_path=str(req_path),
        output_dir=str(out_dir),
        result_path=str(result_path),
        returncode=proc.returncode,
        elapsed_sec=elapsed,
        stdout_tail=proc.stdout[-4000:],
        stderr_tail=proc.stderr[-4000:],
        error=error,
        payload=payload,
    )



class NemoASRWarmClient:
    def __init__(
        self,
        request: NemoASRRequest,
        *,
        popen_factory: Callable[..., subprocess.Popen[Any]] | None = None,
        cwd: str | Path = REPO_ROOT,
    ) -> None:
        self.request = request
        self.output_dir = Path(request.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_path = write_nemo_asr_request(request, self.output_dir)
        self.command = build_nemo_asr_service_command(self.request_path)
        self._popen_factory = popen_factory or subprocess.Popen
        self._cwd = Path(cwd)
        self.process: subprocess.Popen[Any] | None = None
        self.startup_payload: dict[str, Any] | None = None

    def start(self, *, timeout_sec: float | None = None) -> dict[str, Any]:
        if self.process is not None:
            return self.startup_payload or {"status": "already_started"}
        self.process = self._popen_factory(
            self.command,
            cwd=self._cwd,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        started = time.monotonic()
        line = self._readline(timeout_sec, started)
        payload = self._decode_line(line)
        self.startup_payload = payload
        if payload.get("status") not in {"loaded", "ready"}:
            raise NemoASRServiceError(f"NeMo ASR worker failed to start: {payload}")
        return payload

    def transcribe(self, audio_paths: list[str] | tuple[str, ...], *, timeout_sec: float | None = None) -> dict[str, Any]:
        self.start(timeout_sec=timeout_sec)
        assert self.process is not None
        if self.process.stdin is None:
            raise NemoASRServiceError("NeMo ASR worker stdin is not available")
        self.process.stdin.write(json.dumps({"action": "transcribe", "audio_paths": list(audio_paths)}) + "\n")
        self.process.stdin.flush()
        return self._decode_line(self._readline(timeout_sec, time.monotonic()))

    def status(self, *, timeout_sec: float | None = None) -> dict[str, Any]:
        self.start(timeout_sec=timeout_sec)
        assert self.process is not None
        if self.process.stdin is None:
            raise NemoASRServiceError("NeMo ASR worker stdin is not available")
        self.process.stdin.write(json.dumps({"action": "status"}) + "\n")
        self.process.stdin.flush()
        return self._decode_line(self._readline(timeout_sec, time.monotonic()))

    def close(self, *, timeout_sec: float = 10.0) -> dict[str, Any] | None:
        if self.process is None:
            return None
        payload = None
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                self.process.stdin.flush()
                payload = self._decode_line(self._readline(timeout_sec, time.monotonic()))
            except (BrokenPipeError, NemoASRServiceError):
                payload = None
        try:
            self.process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout_sec)
        finally:
            self.process = None
        return payload

    def __enter__(self) -> "NemoASRWarmClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _readline(self, timeout_sec: float | None, started: float) -> str:
        if self.process is None or self.process.stdout is None:
            raise NemoASRServiceError("NeMo ASR worker stdout is not available")
        skipped: list[str] = []
        stdout = self.process.stdout
        if isinstance(stdout, io.StringIO):
            return self._readline_blocking(stdout, timeout_sec, started, skipped)
        try:
            fileno = stdout.fileno()
        except (AttributeError, io.UnsupportedOperation):
            return self._readline_blocking(stdout, timeout_sec, started, skipped)

        with selectors.DefaultSelector() as selector:
            selector.register(fileno, selectors.EVENT_READ)
            while True:
                remaining = None
                if timeout_sec is not None:
                    elapsed = time.monotonic() - started
                    remaining = max(0.0, timeout_sec - elapsed)
                    if remaining <= 0:
                        raise NemoASRServiceError("timed out waiting for NeMo ASR worker")
                events = selector.select(remaining)
                if not events:
                    raise NemoASRServiceError("timed out waiting for NeMo ASR worker")
                line = stdout.readline()
                if not line:
                    self._raise_exited_before_response(skipped)
                if not line.strip():
                    continue
                if not line.lstrip().startswith("{"):
                    skipped.append(line)
                    continue
                return line

    def _readline_blocking(self, stdout: TextIO, timeout_sec: float | None, started: float, skipped: list[str]) -> str:
        while True:
            if timeout_sec is not None and time.monotonic() - started > timeout_sec:
                raise NemoASRServiceError("timed out waiting for NeMo ASR worker")
            line = stdout.readline()
            if not line:
                self._raise_exited_before_response(skipped)
            if not line.strip():
                continue
            if not line.lstrip().startswith("{"):
                skipped.append(line)
                continue
            return line

    def _raise_exited_before_response(self, skipped: list[str]) -> None:
        stderr = ""
        if self.process is not None and self.process.stderr is not None:
            try:
                stderr = self.process.stderr.read()[-4000:]
            except Exception:
                stderr = ""
        skipped_tail = "".join(skipped)[-1000:]
        raise NemoASRServiceError(f"NeMo ASR worker exited before response: {stderr or skipped_tail}")

    @staticmethod
    def _decode_line(line: str) -> dict[str, Any]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NemoASRServiceError(f"NeMo ASR worker returned invalid JSON: {line[:500]}") from exc
        if not isinstance(payload, dict):
            raise NemoASRServiceError("NeMo ASR worker returned a non-object JSON response")
        return payload


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
