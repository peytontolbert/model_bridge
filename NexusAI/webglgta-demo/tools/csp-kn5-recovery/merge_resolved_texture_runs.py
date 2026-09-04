#!/usr/bin/env python3
'''Merge a broad CSP texture run with a protected-texture overlay run.'''

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

VIRTUAL_REFERENCES = {'CSP/<placeholder>', 'CSP/<color>'}


def load_manifest(run: Path) -> dict:
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'CSP_RESOLVED_TEXTURE_RECOVERY_1':
        raise RuntimeError(f'Unsupported manifest: {manifest_path}')
    if manifest.get('track') != 'nohesi_110':
        raise RuntimeError(f'Unexpected track in {manifest_path}')
    return manifest


def safe_name(value: str) -> str:
    leaf = value.replace('\\', '/').rsplit('/', 1)[-1]
    leaf = leaf.replace('::null.dds', '')
    cleaned = ''.join(c if c.isalnum() or c in '._-' else '_' for c in leaf)
    return cleaned.strip('_')[:72] or 'texture'


def link_capture(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f'Missing source DDS: {source}')
    with source.open('rb') as stream:
        if stream.read(4) != b'DDS ':
            raise RuntimeError(f'Invalid source DDS: {source}')
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def merge(base_run: Path, overlay_run: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f'Refusing to replace existing output: {output}')
    base = load_manifest(base_run)
    overlay = load_manifest(overlay_run)
    if overlay.get('capturedCount') != 17 or overlay.get('unresolvedCount') != 0:
        raise RuntimeError('Protected overlay is not a complete 17/17 capture')
    overlay_by_ref = {
        record['runtimeReference']: record
        for record in overlay['textures']
        if record.get('status') == 'captured'
    }
    output.mkdir(parents=True)
    merged_records: list[dict] = []
    excluded_virtual: list[str] = []
    overlaid = 0
    try:
        for base_record in base['textures']:
            reference = base_record['runtimeReference']
            if reference in VIRTUAL_REFERENCES:
                excluded_virtual.append(reference)
                continue
            source_run = base_run
            selected = base_record
            if selected.get('status') != 'captured':
                selected = overlay_by_ref.get(reference)
                source_run = overlay_run
                if selected is None:
                    raise RuntimeError(f'Still unresolved: {reference}')
                overlaid += 1
            source_file = selected.get('outputFile')
            if not source_file or Path(source_file).name != source_file:
                raise RuntimeError(f'Unsafe output filename for {reference}')
            source = source_run / source_file
            merged_id = len(merged_records) + 1
            filename = f'resolved_{merged_id:04d}_{safe_name(reference)}.dds'
            destination = output / filename
            link_capture(source, destination)
            record = dict(selected)
            record['id'] = merged_id
            record['outputFile'] = filename
            record['status'] = 'captured'
            record['byteLength'] = destination.stat().st_size
            if base_record.get('bindings'):
                record['bindings'] = base_record['bindings']
            merged_records.append(record)
        manifest = {
            'format': 'CSP_RESOLVED_TEXTURE_RECOVERY_1',
            'revision': 8,
            'track': 'nohesi_110',
            'trackFullID': base.get('trackFullID'),
            'cspVersion': overlay.get('cspVersion'),
            'source': 'merged broad capture plus CSP loader-native protected material recovery',
            'sourceRuns': [base_run.name, overlay_run.name],
            'strictNoFallback': True,
            'protectedOverlayCount': overlaid,
            'excludedVirtualReferences': excluded_virtual,
            'textureCount': len(merged_records),
            'capturedCount': len(merged_records),
            'unresolvedCount': 0,
            'textures': merged_records,
        }
        (output / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        (output / 'DONE.txt').write_text(
            '110 merged CSP texture recovery\n'
            f'Captured file-backed textures: {len(merged_records)}\n'
            f'Protected loader-native overlays: {overlaid}\n'
            f'Excluded CSP virtual references: {len(excluded_virtual)}\n',
            encoding='utf-8',
        )
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--overlay-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = merge(
        args.base_run.resolve(), args.overlay_run.resolve(), args.output.resolve()
    )
    captured = manifest['capturedCount']
    overlaid = manifest['protectedOverlayCount']
    print(
        f'Merged {captured} textures, including '
        f'{overlaid} protected overlays'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
