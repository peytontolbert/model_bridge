#!/usr/bin/env python3
"""Finalize CSP textures with decoded DDS dimensions and resumable cache."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from finalize_resolved_textures import load_manifest


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compile_record(run, staging, cache, record, maximum):
    relative = record.get('outputFile')
    if not relative or Path(relative).name != relative:
        raise RuntimeError('Unsafe texture output name')
    source = run / relative
    if not source.is_file() or source.stat().st_size != record.get('byteLength'):
        raise RuntimeError('Missing or truncated DDS: {}'.format(source))
    with source.open('rb') as stream:
        if stream.read(4) != b'DDS ':
            raise RuntimeError('Invalid DDS signature: {}'.format(source))
    digest = sha256(source)
    web_name = '{:04d}_{}.webp'.format(int(record['id']), digest)
    destination = staging / 'textures' / web_name
    cached = cache / 'textures' / web_name
    with Image.open(source) as header:
        source_size = header.size
    if min(source_size) <= 1:
        raise RuntimeError('Rejected placeholder-sized DDS: {}'.format(source))
    if cached.is_file():
        with Image.open(cached) as encoded:
            encoded.load()
            web_size = encoded.size
        if min(web_size) <= 1 or max(web_size) > maximum:
            raise RuntimeError('Invalid cached WebP: {}'.format(cached))
        shutil.copy2(cached, destination)
    else:
        with Image.open(source) as image:
            image.load()
            if image.format != 'DDS' or image.size != source_size:
                raise RuntimeError('DDS decode mismatch: {}'.format(source))
            if max(image.size) > maximum:
                image.thumbnail((maximum, maximum), Image.Resampling.BOX)
            web_size = image.size
            image.save(destination, 'WEBP', quality=92, method=2)
        with Image.open(destination) as encoded:
            encoded.load()
            if encoded.size != web_size or min(encoded.size) <= 1:
                raise RuntimeError('WebP verification failed: {}'.format(destination))
    return {
        'id': record['id'], 'runtimeReference': record['runtimeReference'],
        'resolvedImageSource': record.get('resolvedImageSource'),
        'width': source_size[0], 'height': source_size[1],
        'reportedWidth': record.get('width'), 'reportedHeight': record.get('height'),
        'webWidth': web_size[0], 'webHeight': web_size[1],
        'sourceSha256': digest, 'webPath': 'textures/{}'.format(web_name),
        'bindings': record.get('bindings', []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--max-web-dimension', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    run, output, cache = args.run.resolve(), args.output.resolve(), args.cache.resolve()
    manifest = load_manifest(run, 'nohesi_110')
    if manifest.get('capturedCount') != len(manifest.get('textures', [])) or manifest.get('unresolvedCount'):
        raise RuntimeError('CSP texture recovery is incomplete')
    staging = output.with_name(output.name + '.staging')
    if output.exists() or staging.exists():
        raise RuntimeError('Output or staging path already exists')
    (staging / 'textures').mkdir(parents=True)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(compile_record, run, staging, cache, record, args.max_web_dimension)
                for record in manifest['textures']
            ]
            records = [future.result() for future in futures]
        web_manifest = {
            'format': 'CSP_RESOLVED_TEXTURE_WEB_1', 'track': manifest['track'],
            'strictNoFallback': True, 'dimensionAuthority': 'decoded DDS header',
            'maxWebDimension': args.max_web_dimension,
            'webEncoding': {'format': 'webp', 'quality': 92, 'method': 2},
            'sourceRun': run.name, 'textureCount': len(records),
            'uniquePayloadCount': len({record['sourceSha256'] for record in records}),
            'textures': records,
        }
        (staging / 'manifest.json').write_text(json.dumps(web_manifest, indent=2) + '\n', encoding='utf-8')
        staging.rename(output)
        print(json.dumps({'textures': len(records), 'unique': web_manifest['uniquePayloadCount'], 'status': 'complete'}, indent=2))
    except Exception:
        print('Staging preserved for diagnosis/resume: {}'.format(staging))
        raise


if __name__ == '__main__':
    main()
