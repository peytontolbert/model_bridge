#!/usr/bin/env python3
"""Validate CSP texture recovery output and produce browser-ready WebP assets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image


DEFAULT_RUNS_ROOT = Path(r"K:\LANParty\recovered\nohesi_110\resolved_texture_runs")
DEFAULT_MAX_WEB_DIMENSION = 1024


def latest_run(root: Path) -> Path:
    runs = sorted(
        (path for path in root.glob("recovery_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise RuntimeError(f"No CSP texture recovery runs found in {root}")
    return runs[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(run: Path, expected_track: str | None) -> dict:
    path = run / "manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Missing recovery manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "CSP_RESOLVED_TEXTURE_RECOVERY_1":
        raise RuntimeError("Unsupported CSP recovery manifest format")
    track = manifest.get("track")
    if not isinstance(track, str) or not track:
        raise RuntimeError("Recovery manifest has no track identity")
    if expected_track and track != expected_track:
        raise RuntimeError(
            f"Recovery manifest is for {track}, expected {expected_track}"
        )
    if manifest.get("strictNoFallback") is not True:
        raise RuntimeError("Recovery manifest does not assert strict no-fallback mode")
    return manifest


def validate_capture(run: Path, record: dict) -> tuple[Path, Image.Image]:
    relative = record.get("outputFile")
    if not relative or Path(relative).name != relative:
        raise RuntimeError(f"Unsafe or missing outputFile for texture {record.get('id')}")
    source = run / relative
    if not source.is_file():
        raise RuntimeError(f"Missing captured DDS: {source}")
    if source.stat().st_size != record.get("byteLength"):
        raise RuntimeError(f"Byte-length mismatch: {source}")
    if source.read_bytes()[:4] != b"DDS ":
        raise RuntimeError(f"Invalid DDS signature: {source}")

    image = Image.open(source)
    image.load()
    if image.format != "DDS":
        raise RuntimeError(f"Pillow did not decode {source} as DDS")
    width, height = image.size
    if width <= 1 or height <= 1:
        raise RuntimeError(f"Rejected placeholder-sized image: {source} ({width}x{height})")
    if [width, height] != [record.get("width"), record.get("height")]:
        raise RuntimeError(f"Dimension mismatch: {source}")
    return source, image


def compile_web_record(
    run: Path, staging: Path, record: dict, max_web_dimension: int
) -> dict:
    source, image = validate_capture(run, record)
    try:
        source_hash = sha256(source)
        if max(image.size) > max_web_dimension:
            image.thumbnail(
                (max_web_dimension, max_web_dimension),
                Image.Resampling.LANCZOS,
            )
        web_size = image.size
        web_name = f"{record['id']:04d}_{source_hash}.webp"
        destination = staging / "textures" / web_name
        image.save(destination, "WEBP", quality=92, method=2)
        with Image.open(destination) as encoded:
            encoded.load()
            if encoded.size != image.size or min(encoded.size) <= 1:
                raise RuntimeError(f"WebP verification failed: {destination}")
        return {
            "id": record["id"],
            "runtimeReference": record["runtimeReference"],
            "resolvedImageSource": record.get("resolvedImageSource"),
            "width": record["width"],
            "height": record["height"],
            "webWidth": web_size[0],
            "webHeight": web_size[1],
            "sourceSha256": source_hash,
            "webPath": f"textures/{web_name}",
            "bindings": record.get("bindings", []),
        }
    finally:
        image.close()


def compile_or_resume_web_record(
    run: Path, staging: Path, record: dict, max_web_dimension: int
) -> dict:
    cache = staging.with_name(staging.name.replace('.staging', '.resume-cache')) / 'textures'
    relative = record.get('outputFile')
    if relative and Path(relative).name == relative:
        source = run / relative
        if source.is_file() and source.stat().st_size == record.get('byteLength'):
            source_hash = sha256(source)
            web_name = '{:04d}_{}.webp'.format(record['id'], source_hash)
            cached = cache / web_name
            if cached.is_file():
                with Image.open(cached) as encoded:
                    encoded.load()
                    web_size = encoded.size
                if min(web_size) > 1 and max(web_size) <= max_web_dimension:
                    shutil.copy2(cached, staging / 'textures' / web_name)
                    return {
                        'id': record['id'],
                        'runtimeReference': record['runtimeReference'],
                        'resolvedImageSource': record.get('resolvedImageSource'),
                        'width': record['width'],
                        'height': record['height'],
                        'webWidth': web_size[0],
                        'webHeight': web_size[1],
                        'sourceSha256': source_hash,
                        'webPath': 'textures/{}'.format(web_name),
                        'bindings': record.get('bindings', []),
                    }
    return compile_web_record(run, staging, record, max_web_dimension)


def finalize(
    run: Path,
    output: Path,
    expected_track: str | None,
    max_web_dimension: int,
) -> dict:
    manifest = load_manifest(run, expected_track)
    unresolved = [
        record for record in manifest.get("textures", [])
        if record.get("status") != "captured"
    ]
    if unresolved or manifest.get("unresolvedCount", 0) != 0:
        names = ", ".join(
            str(record.get("runtimeReference", record.get("id")))
            for record in unresolved[:8]
        )
        raise RuntimeError(
            f"Strict recovery is incomplete ({len(unresolved)} unresolved): {names}"
        )
    if not (run / "DONE.txt").is_file() or (run / "INCOMPLETE.txt").exists():
        raise RuntimeError("Run is not marked complete")

    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / "textures").mkdir()

    try:
        workers = min(4, max(1, len(manifest["textures"])))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            web_records = list(
                executor.map(
                    lambda record: compile_or_resume_web_record(
                        run, staging, record, max_web_dimension
                    ),
                    manifest["textures"],
                )
            )
        unique_hashes = {record["sourceSha256"] for record in web_records}

        web_manifest = {
            "format": "CSP_RESOLVED_TEXTURE_WEB_1",
            "track": manifest["track"],
            "strictNoFallback": True,
            "maxWebDimension": max_web_dimension,
            "webEncoding": {"format": "webp", "quality": 92, "method": 2},
            "sourceRun": run.name,
            "textureCount": len(web_records),
            "uniquePayloadCount": len(unique_hashes),
            "textures": web_records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(web_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if output.exists():
            raise RuntimeError(f"Refusing to replace existing output: {output}")
        staging.rename(output)
        return web_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, help="Recovery run directory; defaults to latest")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, help="Output directory; defaults to <run>/web")
    parser.add_argument("--expected-track", help="Reject a recovery run for a different track")
    parser.add_argument(
        "--max-web-dimension",
        type=int,
        default=DEFAULT_MAX_WEB_DIMENSION,
        help="Maximum width or height of browser textures",
    )
    args = parser.parse_args()

    try:
        run = args.run.resolve() if args.run else latest_run(args.runs_root.resolve())
        output = args.output.resolve() if args.output else run / "web"
        if args.max_web_dimension < 2:
            raise RuntimeError("--max-web-dimension must be at least 2")
        result = finalize(
            run,
            output,
            args.expected_track,
            args.max_web_dimension,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"Validated and converted {result['textureCount']} textures "
        f"({result['uniquePayloadCount']} unique payloads) into {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
