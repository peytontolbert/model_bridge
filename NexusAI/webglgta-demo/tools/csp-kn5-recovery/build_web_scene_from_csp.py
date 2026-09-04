#!/usr/bin/env python3
"""Compile CSP_SCENE_RECOVERY_1 component batches into streamed TNM v4 tiles.

The compiler treats CSP's live SceneReference geometry and loader-resolved
texture binding manifest as authoritative.  It does not parse protected KN5
payloads or substitute placeholder geometry/textures.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TNM_HEADER = struct.Struct('<4sIIII6f')
NONVISUAL_COMPONENTS = {'pits', 'tunnel_nodes', 'collider'}
SLOT_CHANNELS = {
    'txdiffuse': 'diffuse',
    'txnormal': 'normal',
    'txdetail': 'detail',
    'txdetailnm': 'detailNormal',
    'txnormaldetail': 'normalDetail',
    'txmaps': 'maps',
    'txmask': 'mask',
    'txvariation': 'variation',
    'txdetailr': 'detailR',
    'txdetailg': 'detailG',
    'txdetailb': 'detailB',
    'txdetaila': 'detailA',
    'txemissive': 'emissive',
}


def clean_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_') or 'component'


def source_key(value: Any) -> str:
    return Path(str(value or '').replace('\\', '/').split('#', 1)[0]).name.lower()


def material_key(source: Any, material: Any, shader: Any) -> tuple[str, str, str]:
    return source_key(source), str(material or '').casefold(), str(shader or '').casefold()


def load_old_materials(scene_path: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not scene_path:
        return {}
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model in scene.get('models', []):
        source = model.get('source') or model.get('tileOf') or model.get('file')
        for group in model.get('groups') or []:
            key = material_key(source, group.get('material'), group.get('shader'))
            result.setdefault(key, group)
    return result


def finite_number(value: Any, fallback: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback


def bounded_property(props: dict[str, Any], name: str, fallback: float, low: float, high: float) -> float:
    value = finite_number(props.get(name), fallback)
    return value if low <= value <= high else fallback


def safe_properties(old: dict[str, Any] | None, emissive: bool) -> tuple[dict[str, float], dict[str, Any]]:
    raw = old.get('properties', {}) if isinstance(old, dict) else {}
    props = {
        'ksambient': bounded_property(raw, 'ksambient', 0.34, 0.0, 2.0),
        'ksdiffuse': bounded_property(raw, 'ksdiffuse', 0.66, 0.0, 2.0),
        'ksspecular': bounded_property(raw, 'ksspecular', 0.08, 0.0, 4.0),
        'ksspecularexp': bounded_property(raw, 'ksspecularexp', 18.0, 1.0, 512.0),
        'ksemissive': bounded_property(raw, 'ksemissive', 1.0 if emissive else 0.0, 0.0, 16.0),
        'ksalpharef': bounded_property(raw, 'ksalpharef', 0.5, 0.0, 1.0),
        'magicmult': bounded_property(raw, 'magicmult', 1.0, 0.0, 4.0),
        'fresnelc': bounded_property(raw, 'fresnelc', 0.0, 0.0, 4.0),
        'fresnelexp': bounded_property(raw, 'fresnelexp', 1.0, 0.01, 32.0),
        'fresnelmaxlevel': bounded_property(raw, 'fresnelmaxlevel', 0.0, 0.0, 4.0),
        'tarmacspecularmultiplier': bounded_property(raw, 'tarmacspecularmultiplier', 1.0, 0.0, 16.0),
        'detailuvmultiplier': bounded_property(raw, 'detailuvmultiplier', 1.0, 0.01, 1000.0),
        'usedetail': bounded_property(raw, 'usedetail', 1.0, 0.0, 1.0),
        'detailnormalblend': bounded_property(raw, 'detailnormalblend', 1.0, 0.0, 8.0),
    }
    vectors = old.get('propertyVectors', {}) if isinstance(old, dict) else {}
    safe_vectors = {k: v for k, v in vectors.items() if isinstance(v, list) and all(math.isfinite(finite_number(x, float('nan'))) for x in v)}
    return props, safe_vectors


def texture_bindings(web_manifest: Path) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[tuple[str, str], dict[str, str]], int]:
    manifest = json.loads(web_manifest.read_text(encoding='utf-8'))
    records = manifest.get('textures', [])
    if manifest.get('textureCount') != len(records):
        raise RuntimeError('Browser texture manifest is incomplete')
    if manifest.get('format') == 'CSP_RESOLVED_TEXTURE_RECOVERY_1':
        if manifest.get('unresolvedCount') or manifest.get('capturedCount') != len(records):
            raise RuntimeError('Raw CSP texture recovery is incomplete')
        for texture in records:
            source = web_manifest.parent / str(texture['outputFile'])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            texture['webPath'] = 'textures/{:04d}_{}.webp'.format(int(texture['id']), digest)
    exact: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    fallback: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for texture in records:
        web_path = str(texture['webPath']).replace('\\', '/')
        for binding in texture.get('bindings', []):
            channel = SLOT_CHANNELS.get(str(binding.get('slot') or '').casefold())
            if not channel:
                continue
            key = material_key(binding.get('sourceFile'), binding.get('material'), binding.get('shader'))
            exact[key][channel] = web_path
            fallback[(key[0], key[1])][channel] = web_path
    return exact, fallback, len(records)


def transform_stream(vertices: np.ndarray, transform: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    side = np.asarray(transform.get('side', [1, 0, 0]), dtype=np.float32)
    up = np.asarray(transform.get('up', [0, 1, 0]), dtype=np.float32)
    look = np.asarray(transform.get('look', [0, 0, 1]), dtype=np.float32)
    position = np.asarray(transform.get('position', [0, 0, 0]), dtype=np.float32)
    basis = np.stack((side, up, look), axis=1)
    source_pos = vertices[:, :3] @ basis.T + position
    source_normal = vertices[:, 3:6] @ basis.T
    source_tangent = vertices[:, 8:11] @ basis.T
    positions = source_pos[:, [0, 2, 1]].astype(np.float32, copy=False)
    normals = source_normal[:, [0, 2, 1]]
    tangents = source_tangent[:, [0, 2, 1]]
    for stream in (normals, tangents):
        length = np.linalg.norm(stream, axis=1, keepdims=True)
        stream /= np.maximum(length, 1e-12)
    return positions, normals, vertices[:, 6:8].astype(np.float32, copy=False), tangents


def packed_direction(stream: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(stream, -1.0, 1.0) * 127.0).astype(np.int8)


def append_tile(tile: dict[str, Any], positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray,
                tangents: np.ndarray, triangles: np.ndarray, group_key: tuple[str, str, str], node: str) -> None:
    used, inverse = np.unique(triangles.reshape(-1), return_inverse=True)
    base = tile['vertexCount']
    tile['positions'].append(np.ascontiguousarray(positions[used], dtype='<f4'))
    tile['normals'].append(packed_direction(normals[used]))
    tile['tangents'].append(packed_direction(tangents[used]))
    tile['uvs'].append(np.ascontiguousarray(uvs[used], dtype='<f4'))
    indices = inverse.astype(np.uint32) + np.uint32(base)
    tile['indices'][group_key].append(indices.astype('<u4', copy=False))
    tile['nodes'][group_key].add(node)
    tile['vertexCount'] += len(used)
    tile['triangleCount'] += len(triangles)


def alpha_mode(shader: str, material: str, transparent: bool) -> str:
    key = f'{shader} {material}'.lower()
    if transparent or 'glass' in key or 'translucent' in key:
        return 'blend'
    if 'at_' in key or key.endswith('at') or 'alpha' in key or 'fence' in key or 'decal' in key:
        return 'cutout'
    return 'opaque'


def fallback_color(material: str) -> list[int]:
    digest = hashlib.sha256(material.encode('utf-8', 'replace')).digest()
    return [72 + digest[0] % 72, 72 + digest[1] % 72, 72 + digest[2] % 72]


def write_tile(path: Path, tile: dict[str, Any], material_meta: dict[tuple[str, str, str], dict[str, Any]],
               exact_textures: dict, fallback_textures: dict, old_materials: dict) -> tuple[dict[str, Any], int]:
    positions = np.concatenate(tile['positions'])
    normals = np.concatenate(tile['normals'])
    tangents = np.concatenate(tile['tangents'])
    uvs = np.concatenate(tile['uvs'])
    minimum = positions.min(axis=0).astype(np.float32)
    maximum = positions.max(axis=0).astype(np.float32)
    span = np.maximum(maximum - minimum, np.float32(1e-5))
    group_records = []
    index_streams = []
    offset = 0
    for key in sorted(tile['indices']):
        indices = np.concatenate(tile['indices'][key]).astype('<u4', copy=False)
        meta = material_meta[key]
        textures = dict(fallback_textures.get((key[0], key[1]), {}))
        textures.update(exact_textures.get(key, {}))
        old = old_materials.get(key)
        props, vectors = safe_properties(old, 'emissive' in textures)
        mode = alpha_mode(meta['shader'], meta['material'], meta['transparent'])
        group_records.append({
            'material': meta['material'], 'shader': meta['shader'],
            'offset': offset, 'count': int(len(indices)),
            'color': [255, 255, 255] if 'diffuse' in textures else fallback_color(meta['material']),
            'alphaMode': mode, 'properties': props, 'propertyVectors': vectors,
            'nodeTransparent': meta['transparent'], 'castShadows': meta['castShadows'],
            'sourceNodes': sorted(tile['nodes'][key]), 'textures': {k: f'textures/{Path(v).name}' for k, v in textures.items()},
            'materialBinding': f'csp-{len(group_records):04d}',
        })
        index_streams.append(indices)
        offset += len(indices)
    indices = np.concatenate(index_streams)
    payload = bytearray(TNM_HEADER.pack(b'TNM1', 4, len(positions), len(indices), len(group_records), *minimum, *span))
    payload.extend(positions.astype('<f4', copy=False).tobytes())
    payload.extend(normals.tobytes())
    payload.extend(tangents.tobytes())
    payload.extend(uvs.astype('<f4', copy=False).tobytes())
    payload.extend(indices.tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(bytes(payload), compresslevel=7, mtime=0))
    model = {
        'file': f'tiles/{path.name}', 'source': tile['source'],
        'tileOf': tile['source'], 'tile': {'x': tile['x'], 'y': tile['y']},
        'vertices': int(len(positions)), 'trianglesInput': int(len(indices) // 3),
        'trianglesOutput': int(len(indices) // 3), 'trianglesGeometry': int(len(indices) // 3),
        'trianglesExcludedByNodeFlags': 0, 'groups': group_records,
        'bytes': path.stat().st_size, 'compressedBytes': path.stat().st_size,
        'decodedBytes': len(payload), 'binaryVersion': 4,
        'binaryIndexCount': int(len(indices)), 'binaryGroupCount': len(group_records),
        'bounds': {'min': minimum.tolist(), 'max': maximum.tolist()},
    }
    return model, len(payload)


def new_tile(source: str, x: int, y: int) -> dict[str, Any]:
    return {'source': source, 'x': x, 'y': y, 'positions': [], 'normals': [], 'tangents': [], 'uvs': [],
            'indices': defaultdict(list), 'nodes': defaultdict(set), 'vertexCount': 0, 'triangleCount': 0}


def compile_scene(args: argparse.Namespace) -> None:
    run = args.recovery.resolve()
    aggregate = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if aggregate.get('format') != 'CSP_PER_KN5_RECOVERY_RUN_1' or not aggregate.get('strictNoFallback'):
        raise RuntimeError('Expected a complete strict CSP per-KN5 recovery run')
    if aggregate.get('completedModelCount') != aggregate.get('requestedModelCount'):
        raise RuntimeError('CSP component recovery is incomplete')
    exact_textures, fallback_textures, texture_count = texture_bindings(args.web_textures.resolve())
    old_materials = load_old_materials(args.material_scene.resolve() if args.material_scene else None)
    output = args.output.resolve()
    staging = output.with_name(output.name + '.staging')
    if staging.exists():
        shutil.rmtree(staging)
    (staging / 'tiles').mkdir(parents=True)
    source_texture_dir = args.web_textures.resolve().parent / 'textures'
    if source_texture_dir.exists():
        shutil.copytree(source_texture_dir, staging / 'textures')
    else:
        (staging / 'textures').mkdir()
    material_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    models = []
    total_vertices = total_triangles = decoded_bytes = 0
    global_min = np.full(3, np.inf)
    global_max = np.full(3, -np.inf)
    excluded = {'models': [], 'meshes': 0, 'vertices': 0, 'indices': 0}
    components = aggregate['components']
    for component_index, component in enumerate(components, 1):
        component_name = component['directory']
        component_root = run / component_name
        manifest = json.loads((component_root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('skippedMeshCount') or manifest.get('meshCount') != len(manifest.get('meshes', [])):
            raise RuntimeError(f'Incomplete component: {component_name}')
        if component_name.casefold() in NONVISUAL_COMPONENTS and not args.include_helpers:
            excluded['models'].append(component_name)
            excluded['meshes'] += manifest['meshCount']
            excluded['vertices'] += manifest['vertexCount']
            excluded['indices'] += manifest['indexCount']
            print(f'[{component_index}/{len(components)}] excluded nonvisual {component_name}', flush=True)
            continue
        tiles: dict[tuple[int, int], dict[str, Any]] = {}
        for mesh_index, mesh in enumerate(manifest['meshes'], 1):
            if str(mesh.get('material') or '').casefold() in {'m_collider', 'collider'}:
                excluded['meshes'] += 1
                excluded['vertices'] += int(mesh['vertexCount'])
                excluded['indices'] += int(mesh['indexCount'])
                continue
            batch = component_root / f"batch_{int(mesh['batch']):05d}.bin"
            vertices = np.memmap(batch, dtype='<f4', mode='r', offset=int(mesh['vertexOffset']), shape=(int(mesh['vertexCount']), 11))
            indices = np.memmap(batch, dtype='<u2', mode='r', offset=int(mesh['indexOffset']), shape=(int(mesh['indexCount']),)).astype(np.int64)
            triangles = indices[:len(indices) - len(indices) % 3].reshape(-1, 3)
            valid = (triangles >= 0).all(axis=1) & (triangles < len(vertices)).all(axis=1)
            if not valid.all():
                raise RuntimeError(f'Invalid CSP indices in {component_name}/{mesh["name"]}')
            positions, normals, uvs, tangents = transform_stream(vertices, mesh.get('worldTransform') or {})
            centers = positions[triangles].mean(axis=1)
            tile_xy = np.floor(centers[:, :2] / args.tile_size).astype(np.int32)
            key = material_key(manifest.get('sourceFile'), mesh.get('material'), mesh.get('shader'))
            material_meta.setdefault(key, {'material': str(mesh.get('material') or ''), 'shader': str(mesh.get('shader') or ''),
                                           'transparent': bool(mesh.get('transparent')), 'castShadows': bool(mesh.get('castShadows', True))})
            unique_tiles = np.unique(tile_xy, axis=0)
            for tx, ty in unique_tiles:
                selected = triangles[(tile_xy[:, 0] == tx) & (tile_xy[:, 1] == ty)]
                tile = tiles.setdefault((int(tx), int(ty)), new_tile(manifest['sourceFile'], int(tx), int(ty)))
                append_tile(tile, positions, normals, uvs, tangents, selected, key, str(mesh.get('name') or ''))
            del vertices, indices, triangles, positions, normals, uvs, tangents, centers, tile_xy
        safe_component = clean_name(component_name)
        component_triangles = 0
        for (tx, ty), tile in sorted(tiles.items()):
            name = f'{safe_component}_{tx}_{ty}.tnm.gz'
            model, raw_bytes = write_tile(staging / 'tiles' / name, tile, material_meta, exact_textures,
                                          fallback_textures, old_materials)
            models.append(model)
            component_triangles += model['trianglesOutput']
            total_vertices += model['vertices']
            total_triangles += model['trianglesOutput']
            decoded_bytes += raw_bytes
            global_min = np.minimum(global_min, np.asarray(model['bounds']['min']))
            global_max = np.maximum(global_max, np.asarray(model['bounds']['max']))
        print(f'[{component_index}/{len(components)}] {component_name}: {len(tiles)} tiles, {component_triangles:,} triangles', flush=True)
    expected_rendered = aggregate['indexCount'] - excluded['indices']
    if total_triangles * 3 != expected_rendered:
        raise RuntimeError(f'Index conservation failure: emitted {total_triangles * 3}, expected {expected_rendered}')
    scene = {
        'schema': 'webglgta-track-scene-v1', 'id': 'nohesi-110-csp-full',
        'sourceFormat': aggregate['format'], 'source': 'Assetto Corsa CSP SceneReference loader recovery',
        'coordinateSystem': 'demo-data-x-y-z-up',
        'compression': {'positions': 'float32 absolute', 'indices': 'uint32', 'transport': 'gzip',
                        'geometrySnapM': 0, 'spatialTileM': args.tile_size},
        'bounds': {'minX': float(global_min[0]), 'minY': float(global_min[1]), 'minZ': float(global_min[2]),
                   'maxX': float(global_max[0]), 'maxY': float(global_max[1]), 'maxZ': float(global_max[2])},
        'models': models,
        'textures': {'format': 'CSP_WEB_TEXTURE_SET_1', 'manifest': 'textures-manifest.json'},
        'audit': {
            'cspRequestedModels': aggregate['requestedModelCount'], 'cspCompletedModels': aggregate['completedModelCount'],
            'cspMeshCount': aggregate['meshCount'], 'cspVertexCount': aggregate['vertexCount'],
            'cspIndexCount': aggregate['indexCount'], 'renderedModelCount': len(models),
            'renderedVertexCount': total_vertices, 'renderedTriangleCount': total_triangles,
            'nonvisualExcluded': excluded, 'textureCount': texture_count, 'strictNoFallback': True,
        },
        'modelCount': len(models), 'totalTriangles': total_triangles, 'decodedBytes': decoded_bytes,
    }
    shutil.copy2(args.web_textures.resolve(), staging / 'textures-manifest.json')
    (staging / 'scene.json').write_text(json.dumps(scene, separators=(',', ':')), encoding='utf-8')
    (staging / 'EXPORT_AUDIT.json').write_text(json.dumps(scene['audit'], indent=2) + '\n', encoding='utf-8')
    if output.exists():
        backup = output.with_name(output.name + '.previous')
        if backup.exists():
            shutil.rmtree(backup)
        output.rename(backup)
    staging.rename(output)
    print(json.dumps(scene['audit'] | {'bounds': scene['bounds'], 'decodedBytes': decoded_bytes}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--web-textures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--material-scene', type=Path)
    parser.add_argument('--tile-size', type=int, default=384)
    parser.add_argument('--include-helpers', action='store_true')
    compile_scene(parser.parse_args())


if __name__ == '__main__':
    main()
