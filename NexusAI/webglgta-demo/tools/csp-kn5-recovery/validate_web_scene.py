#!/usr/bin/env python3
"""Validate every TNM tile and all manifest/file/count invariants."""

import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct('<4sIIII6f')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=Path)
    parser.add_argument('--require-textures', action='store_true')
    args = parser.parse_args()
    scene_path = args.scene.resolve()
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    triangle_count = vertex_count = group_count = decoded_bytes = 0
    texture_refs = set()
    missing_textures = set()
    for number, model in enumerate(scene.get('models', []), 1):
        tile_path = scene_path.parent / model['file']
        raw = gzip.decompress(tile_path.read_bytes())
        if len(raw) < HEADER.size:
            raise RuntimeError(f'Truncated tile: {tile_path}')
        magic, version, vertices, indices, groups, *bounds = HEADER.unpack_from(raw)
        if magic != b'TNM1' or version != 4 or indices % 3:
            raise RuntimeError(f'Invalid TNM header: {tile_path}')
        expected = HEADER.size + vertices * 12 + vertices * 3 + vertices * 3 + vertices * 8 + indices * 4
        if len(raw) != expected:
            raise RuntimeError(f'TNM byte mismatch: {tile_path}')
        index_offset = HEADER.size + vertices * 26
        values = np.frombuffer(raw, dtype='<u4', count=indices, offset=index_offset)
        if len(values) and int(values.max()) >= vertices:
            raise RuntimeError(f'Out-of-range TNM index: {tile_path}')
        manifest_groups = model.get('groups', [])
        if groups != len(manifest_groups) or sum(int(group['count']) for group in manifest_groups) != indices:
            raise RuntimeError(f'Group/index mismatch: {tile_path}')
        if model.get('vertices') != vertices or model.get('trianglesOutput') != indices // 3:
            raise RuntimeError(f'Manifest/header count mismatch: {tile_path}')
        for group in manifest_groups:
            for value in (group.get('textures') or {}).values():
                texture_refs.add(value)
                if not (scene_path.parent / value).is_file():
                    missing_textures.add(value)
        triangle_count += indices // 3
        vertex_count += vertices
        group_count += groups
        decoded_bytes += len(raw)
        if number % 500 == 0:
            print(f'validated {number}/{len(scene["models"])} tiles', flush=True)
    audit = scene.get('audit', {})
    if triangle_count != audit.get('renderedTriangleCount') or vertex_count != audit.get('renderedVertexCount'):
        raise RuntimeError('Scene audit totals do not match TNM payloads')
    if args.require_textures and missing_textures:
        raise RuntimeError(f'{len(missing_textures)} referenced textures are missing')
    print(json.dumps({
        'tiles': len(scene.get('models', [])), 'groups': group_count,
        'vertices': vertex_count, 'triangles': triangle_count,
        'decodedBytes': decoded_bytes, 'textureReferences': len(texture_refs),
        'missingTextures': len(missing_textures), 'status': 'valid',
    }, indent=2))


if __name__ == '__main__':
    main()
