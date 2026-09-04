#!/usr/bin/env python3
"""Compile a lightweight whole-map overview from a recovered TNM v4 scene.

The source scene remains the authoritative CSP SceneReference export. This pass
only creates a display LOD: it decimates each material partition, merges the
384 m source tiles into larger cells, replaces texture sampling with an average
material colour, and omits distant 2D scenery from the playable-map bounds.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import re
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import fast_simplification
import numpy as np
from PIL import Image


TNM_HEADER = struct.Struct('<4sIIII6f')
ROAD_WORDS = re.compile(r'road|track|asphalt|tarmac|lane|highway|bridge|tunnel|kerb|curb|sidewalk|pavement|concrete|ground|terrain|parking', re.I)
ASPHALT_WORDS = re.compile(r'road|track|asphalt|tarmac|lane|highway|bridge|tunnel|parking', re.I)
GROUND_WORDS = re.compile(r'terrain|ground|grass|landscape|filler_ground', re.I)
MARKING_WORDS = re.compile(r'kerb|curb|line|marking|road_mark|decal', re.I)
STRUCTURE_WORDS = re.compile(r'building|urban|wall|roof|house|garage|industrial|barrier', re.I)
FOLIAGE_WORDS = re.compile(r'tree|bush|foliage|leaf|leaves|vegetation|hedge|grass', re.I)
DISTANT_SCENERY_WORDS = re.compile(r'mountain_2d|background_cards|terrain_horizon|skybox', re.I)


def decode_tnm(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < TNM_HEADER.size:
        raise ValueError(f'truncated TNM: {path}')
    magic, version, vertex_count, index_count, group_count, *_bounds = TNM_HEADER.unpack_from(raw)
    if magic != b'TNM1' or version != 4 or index_count % 3 or group_count < 1:
        raise ValueError(f'unsupported TNM header: {path}')
    cursor = TNM_HEADER.size
    positions = np.frombuffer(raw, '<f4', vertex_count * 3, cursor).reshape(vertex_count, 3).copy()
    cursor += vertex_count * 12
    cursor += vertex_count * 3  # packed normals
    cursor += vertex_count * 3  # packed tangents
    cursor += vertex_count * 8  # UVs
    indices = np.frombuffer(raw, '<u4', index_count, cursor).copy()
    return positions, indices, []


def material_colour(scene_root: Path, group: dict[str, Any], cache: dict[str, list[int]]) -> list[int]:
    label = group_label(group)
    if MARKING_WORDS.search(label):
        return [215, 210, 188]
    if ASPHALT_WORDS.search(label):
        return [112, 120, 126]
    if GROUND_WORDS.search(label):
        return [52, 76, 46]
    if STRUCTURE_WORDS.search(label):
        return [102, 108, 114]
    diffuse = str((group.get('textures') or {}).get('diffuse') or '')
    if not diffuse:
        return [int(max(0, min(255, value))) for value in (group.get('color') or [104, 108, 96])[:3]]
    if diffuse in cache:
        return cache[diffuse]
    path = scene_root / diffuse
    try:
        with Image.open(path) as image:
            image.thumbnail((64, 64), Image.Resampling.BOX)
            pixels = np.asarray(image.convert('RGBA'), dtype=np.float64)
        weights = pixels[..., 3] / 255.0
        total = float(weights.sum())
        if total > 1e-6:
            rgb = (pixels[..., :3] * weights[..., None]).sum(axis=(0, 1)) / total
        else:
            rgb = pixels[..., :3].mean(axis=(0, 1))
        colour = np.rint(np.clip(rgb, 12, 245)).astype(np.uint8).tolist()
    except Exception:
        colour = [int(max(0, min(255, value))) for value in (group.get('color') or [104, 108, 96])[:3]]
    cache[diffuse] = colour
    return colour


def group_label(group: dict[str, Any]) -> str:
    return ' '.join([
        str(group.get('material') or ''),
        str(group.get('shader') or ''),
        *[str(value) for value in (group.get('sourceNodes') or [])],
    ])


def reduction_for(group: dict[str, Any], triangles: int) -> float:
    if triangles < 64:
        return 0.0
    label = group_label(group)
    if ROAD_WORDS.search(label):
        return 0.91
    if STRUCTURE_WORDS.search(label):
        return 0.97
    return 0.985


def overview_group(scene_root: Path, group: dict[str, Any], colour_cache: dict[str, list[int]]) -> dict[str, Any] | None:
    label = group_label(group)
    alpha_mode = str(group.get('alphaMode') or 'opaque')
    if DISTANT_SCENERY_WORDS.search(label):
        return None
    if alpha_mode == 'blend' or (alpha_mode == 'cutout' and FOLIAGE_WORDS.search(label)):
        return None
    properties = group.get('properties') or {}
    emissive = float(properties.get('ksemissive') or 0.0)
    colour = material_colour(scene_root, group, colour_cache)
    if MARKING_WORDS.search(label):
        material_class = 'overview-marking'
    elif ASPHALT_WORDS.search(label):
        material_class = 'overview-road'
    elif GROUND_WORDS.search(label):
        material_class = 'overview-ground'
    elif STRUCTURE_WORDS.search(label):
        material_class = 'overview-structure'
    else:
        colour = [int(round(channel / 32.0) * 32) for channel in colour]
        colour = [max(16, min(240, channel)) for channel in colour]
        material_class = 'overview-colour-' + '-'.join(map(str, colour))
    return {
        'material': material_class,
        'sourceMaterial': str(group.get('material') or 'overview'),
        'shader': 'overview-colour',
        'color': colour,
        'alphaMode': 'opaque',
        'properties': {
            'ksambient': 0.55,
            'ksdiffuse': 0.70,
            'ksspecular': 0.0,
            'ksspecularexp': 18.0,
            'ksemissive': max(0.0, min(0.35, emissive)),
            'ksalpharef': 0.5,
            'magicmult': 1.0,
            'fresnelc': 0.0,
            'fresnelexp': 1.0,
            'fresnelmaxlevel': 0.0,
            'tarmacspecularmultiplier': 0.0,
            'detailuvmultiplier': 1.0,
            'usedetail': 0.0,
            'detailnormalblend': 0.0,
        },
        'propertyVectors': {},
        'nodeTransparent': False,
        'castShadows': False,
        'sourceNodes': [],
        'textures': {},
    }


def group_key(group: dict[str, Any]) -> str:
    stable = {key: value for key, value in group.items() if key not in {'offset', 'count', 'materialBinding'}}
    return json.dumps(stable, sort_keys=True, separators=(',', ':'))


def simplify_partition(positions: np.ndarray, faces: np.ndarray, reduction: float,
                       weld_grid: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    local_positions = np.ascontiguousarray(positions[used], dtype=np.float64)
    local_faces = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32)
    if weld_grid > 0.0:
        weld_keys = np.rint(local_positions / weld_grid).astype(np.int64)
        _unique, first, welded = np.unique(
            weld_keys, axis=0, return_index=True, return_inverse=True,
        )
        local_positions = np.ascontiguousarray(local_positions[first], dtype=np.float64)
        local_faces = welded[local_faces].astype(np.int32, copy=False)
        valid = (
            (local_faces[:, 0] != local_faces[:, 1])
            & (local_faces[:, 1] != local_faces[:, 2])
            & (local_faces[:, 2] != local_faces[:, 0])
        )
        local_faces = np.ascontiguousarray(local_faces[valid], dtype=np.int32)
    if reduction <= 0.0 or len(local_faces) < 64:
        return local_positions.astype(np.float32), local_faces.astype(np.uint32)
    try:
        reduced_positions, reduced_faces = fast_simplification.simplify(
            local_positions,
            local_faces,
            target_reduction=reduction,
            agg=7.0,
            # The overview is only displayed from kilometres away. Source
            # partitions contain many artificial tile/material borders; making
            # all of them immutable prevents meaningful reduction. The exact
            # border-preserving CSP geometry remains active at close range.
            preserve_border=False,
        )
        if len(reduced_faces) < 1 or len(reduced_positions) < 3:
            raise ValueError('empty simplification result')
        return np.asarray(reduced_positions, dtype=np.float32), np.asarray(reduced_faces, dtype=np.uint32)
    except (RuntimeError, ValueError):
        return local_positions.astype(np.float32), local_faces.astype(np.uint32)


def packed_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float64)
    triangle_normals = np.cross(
        positions[faces[:, 1]] - positions[faces[:, 0]],
        positions[faces[:, 2]] - positions[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], triangle_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    invalid = ~np.isfinite(normals).all(axis=1) | (lengths[:, 0] <= 1e-12)
    normals[invalid] = (0.0, 0.0, 1.0)
    return np.rint(np.clip(normals, -1.0, 1.0) * 127.0).astype(np.int8)


def write_cell(path: Path, cell: dict[str, Any], final_reduction: float) -> tuple[dict[str, Any], int]:
    render_records: dict[str, dict[str, Any]] = {}
    for record in cell['groups'].values():
        positions = np.concatenate(record['positions'])
        faces = np.concatenate(record['faces'])
        # Simplify only within one original recovered material. Combining
        # unrelated materials before QEM can collapse disconnected topology.
        positions, faces = simplify_partition(
            positions, faces, final_reduction, weld_grid=0.10,
        )
        metadata = copy.deepcopy(record['metadata'])
        metadata.pop('sourceMaterial', None)
        key = group_key(metadata)
        target = render_records.setdefault(key, {
            'metadata': metadata, 'positions': [], 'faces': [], 'vertexCount': 0,
        })
        target['positions'].append(positions)
        target['faces'].append(faces + np.uint32(target['vertexCount']))
        target['vertexCount'] += len(positions)

    position_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    groups: list[dict[str, Any]] = []
    vertex_base = 0
    index_offset = 0
    for record in render_records.values():
        positions = np.concatenate(record['positions'])
        faces = np.concatenate(record['faces'])
        normals = packed_normals(positions, faces)
        flat_indices = faces.reshape(-1).astype(np.uint32) + np.uint32(vertex_base)
        group = copy.deepcopy(record['metadata'])
        group['offset'] = index_offset
        group['count'] = int(len(flat_indices))
        group['materialBinding'] = f'overview-{len(groups):04d}'
        groups.append(group)
        position_parts.append(positions.astype('<f4', copy=False))
        normal_parts.append(normals)
        index_parts.append(flat_indices.astype('<u4', copy=False))
        vertex_base += len(positions)
        index_offset += len(flat_indices)
    positions = np.concatenate(position_parts)
    normals = np.concatenate(normal_parts)
    indices = np.concatenate(index_parts)
    minimum = positions.min(axis=0).astype(np.float32)
    maximum = positions.max(axis=0).astype(np.float32)
    span = np.maximum(maximum - minimum, np.float32(1e-5))
    tangents = np.zeros((len(positions), 3), dtype=np.int8)
    tangents[:, 0] = 127
    uvs = np.zeros((len(positions), 2), dtype='<f4')
    payload = bytearray(TNM_HEADER.pack(b'TNM1', 4, len(positions), len(indices), len(groups), *minimum, *span))
    payload.extend(positions.tobytes())
    payload.extend(normals.tobytes())
    payload.extend(tangents.tobytes())
    payload.extend(uvs.tobytes())
    payload.extend(indices.tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(bytes(payload), compresslevel=7, mtime=0))
    model = {
        'file': f'tiles/{path.name}',
        'source': 'nohesi-110-csp-overview',
        'tileOf': 'nohesi-110-csp-overview',
        'tile': {'x': cell['x'], 'y': cell['y']},
        'vertices': int(len(positions)),
        'trianglesInput': int(len(indices) // 3),
        'trianglesOutput': int(len(indices) // 3),
        'trianglesGeometry': int(len(indices) // 3),
        'trianglesExcludedByNodeFlags': 0,
        'groups': groups,
        'bytes': path.stat().st_size,
        'compressedBytes': path.stat().st_size,
        'decodedBytes': len(payload),
        'binaryVersion': 4,
        'binaryIndexCount': int(len(indices)),
        'binaryGroupCount': len(groups),
        'bounds': {'min': minimum.tolist(), 'max': maximum.tolist()},
    }
    return model, len(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell-size', type=float, default=1536.0)
    parser.add_argument('--final-reduction', type=float, default=0.72)
    parser.add_argument('--exclude-source', default=r'mountain_2d|skybox')
    args = parser.parse_args()

    source_scene_path = args.scene.resolve()
    source_root = source_scene_path.parent
    source_scene = json.loads(source_scene_path.read_text(encoding='utf-8'))
    if source_scene.get('schema') != 'webglgta-track-scene-v1' or not source_scene.get('models'):
        raise RuntimeError('Expected a populated recovered track scene')
    exclude_source = re.compile(args.exclude_source, re.I)
    output = args.output.resolve()
    staging = output.with_name(output.name + '.staging')
    if staging.exists():
        shutil.rmtree(staging)
    (staging / 'tiles').mkdir(parents=True)

    cells: dict[tuple[int, int], dict[str, Any]] = {}
    colour_cache: dict[str, list[int]] = {}
    input_triangles = output_triangles = excluded_triangles = 0
    for model_number, model in enumerate(source_scene['models'], 1):
        model_triangles = int(model.get('trianglesOutput') or 0)
        input_triangles += model_triangles
        if exclude_source.search(str(model.get('source') or model.get('tileOf') or '')):
            excluded_triangles += model_triangles
            continue
        positions, indices, _unused = decode_tnm(source_root / str(model['file']))
        bounds = model.get('bounds') or {}
        minimum = np.asarray(bounds.get('min') or positions.min(axis=0), dtype=np.float64)
        maximum = np.asarray(bounds.get('max') or positions.max(axis=0), dtype=np.float64)
        center = (minimum + maximum) * 0.5
        cx, cy = np.floor(center[:2] / args.cell_size).astype(np.int32)
        cell = cells.setdefault((int(cx), int(cy)), {'x': int(cx), 'y': int(cy), 'groups': {}})
        for source_group in model.get('groups') or []:
            offset = int(source_group.get('offset') or 0)
            count = int(source_group.get('count') or 0)
            if count < 3 or offset < 0 or offset + count > len(indices):
                continue
            metadata = overview_group(source_root, source_group, colour_cache)
            if metadata is None:
                excluded_triangles += count // 3
                continue
            faces = indices[offset:offset + count].reshape(-1, 3)
            reduced_positions, reduced_faces = simplify_partition(
                positions,
                faces,
                reduction_for(source_group, len(faces)),
            )
            output_triangles += len(reduced_faces)
            key = group_key(metadata)
            record = cell['groups'].setdefault(key, {'metadata': metadata, 'positions': [], 'faces': []})
            vertex_base = sum(len(part) for part in record['positions'])
            record['positions'].append(reduced_positions)
            record['faces'].append(reduced_faces + np.uint32(vertex_base))
        if model_number % 100 == 0 or model_number == len(source_scene['models']):
            print(json.dumps({
                'processedTiles': model_number,
                'totalTiles': len(source_scene['models']),
                'overviewCells': len(cells),
                'inputTriangles': input_triangles,
                'outputTriangles': output_triangles,
            }), flush=True)

    models: list[dict[str, Any]] = []
    decoded_bytes = 0
    global_min = np.full(3, np.inf)
    global_max = np.full(3, -np.inf)
    for (cx, cy), cell in sorted(cells.items()):
        if not cell['groups']:
            continue
        model, raw_bytes = write_cell(
            staging / 'tiles' / f'overview_{cx}_{cy}.tnm.gz', cell, args.final_reduction,
        )
        models.append(model)
        decoded_bytes += raw_bytes
        global_min = np.minimum(global_min, np.asarray(model['bounds']['min']))
        global_max = np.maximum(global_max, np.asarray(model['bounds']['max']))

    rendered_vertices = sum(int(model['vertices']) for model in models)
    rendered_triangles = sum(int(model['trianglesOutput']) for model in models)
    audit = {
        'sourceScene': source_scene_path.name,
        'sourceSceneId': source_scene.get('id'),
        'sourceModelCount': len(source_scene['models']),
        'sourceTriangleCount': input_triangles,
        'overviewModelCount': len(models),
        'overviewTriangleCount': rendered_triangles,
        'partitionPassTriangleCount': output_triangles,
        'renderedModelCount': len(models),
        'renderedVertexCount': rendered_vertices,
        'renderedTriangleCount': rendered_triangles,
        'excludedTriangleCount': excluded_triangles,
        'reductionRatio': 1.0 - (rendered_triangles / max(1, input_triangles)),
        'cellSizeM': args.cell_size,
        'finalMergedReduction': args.final_reduction,
        'excludeSourcePattern': args.exclude_source,
        'textureMode': 'average-diffuse-material-colour',
    }
    scene = {
        'schema': 'webglgta-track-scene-v1',
        'id': 'nohesi-110-csp-overview-v1',
        'sourceFormat': source_scene.get('sourceFormat'),
        'source': 'Compiled display LOD of Assetto Corsa CSP SceneReference recovery',
        'coordinateSystem': source_scene.get('coordinateSystem'),
        'compression': {
            'positions': 'float32 absolute',
            'indices': 'uint32',
            'transport': 'gzip',
            'geometrySimplification': 'quadric-edge-collapse-overview-display-lod',
            'spatialTileM': args.cell_size,
        },
        'bounds': {
            'minX': float(global_min[0]), 'minY': float(global_min[1]), 'minZ': float(global_min[2]),
            'maxX': float(global_max[0]), 'maxY': float(global_max[1]), 'maxZ': float(global_max[2]),
        },
        'models': models,
        'textures': {'format': 'OVERVIEW_MATERIAL_COLOURS_1', 'count': 0},
        'audit': audit,
        'modelCount': len(models),
        'totalTriangles': rendered_triangles,
        'decodedBytes': decoded_bytes,
    }
    (staging / 'scene.json').write_text(json.dumps(scene, separators=(',', ':')), encoding='utf-8')
    (staging / 'EXPORT_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    if output.exists():
        backup = output.with_name(output.name + '.previous')
        if backup.exists():
            shutil.rmtree(backup)
        output.rename(backup)
    staging.rename(output)
    print(json.dumps(audit | {'bounds': scene['bounds'], 'decodedBytes': decoded_bytes}, indent=2))


if __name__ == '__main__':
    main()
