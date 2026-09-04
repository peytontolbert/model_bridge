#!/usr/bin/env python3
"""Compare Assetto traffic centerlines with recovered road geometry."""

import argparse
import gzip
import json
import re
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HEADER = struct.Struct('<4sIIII6f')
ROAD = re.compile(r'road|track|asphalt|tarmac|lane|highway|bridge|tunnel|kerb|curb|pavement|concrete|parking', re.I)


def decode(path):
    raw = gzip.decompress(path.read_bytes())
    magic, version, vertices, indices, _groups, *_ = HEADER.unpack_from(raw)
    if magic != b'TNM1' or version != 4:
        raise RuntimeError(f'Unsupported TNM file: {path}')
    cursor = HEADER.size
    positions = np.frombuffer(raw, '<f4', vertices * 3, cursor).reshape(vertices, 3)
    cursor += vertices * 26
    return positions, np.frombuffer(raw, '<u4', indices, cursor)


def label(group):
    return ' '.join([
        str(group.get('material') or ''),
        str(group.get('shader') or ''),
        *map(str, group.get('sourceNodes') or []),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--traffic', type=Path, required=True)
    parser.add_argument('--sample-step', type=int, default=20)
    args = parser.parse_args()
    scene_path = args.scene.resolve()
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    traffic = json.loads(args.traffic.read_text(encoding='utf-8'))
    points = np.asarray([
        point for lane in traffic['lanes'] for point in lane['points'][::args.sample_step]
    ], dtype=np.float64)
    clouds = []
    model_count = 0
    for model in scene['models']:
        road_groups = [group for group in model.get('groups') or [] if ROAD.search(label(group))]
        if not road_groups:
            continue
        positions, indices = decode(scene_path.parent / model['file'])
        selected = []
        for group in road_groups:
            offset = int(group.get('offset') or 0)
            count = int(group.get('count') or 0)
            selected.append(indices[offset:offset + count])
        clouds.append(positions[np.unique(np.concatenate(selected))])
        model_count += 1
    road = np.concatenate(clouds).astype(np.float64)
    tree = cKDTree(road[:, :2])
    transforms = {
        'as_imported': lambda p: p,
        'flip_y': lambda p: p * (1, -1, 1),
        'flip_x': lambda p: p * (-1, 1, 1),
        'flip_xy': lambda p: p * (-1, -1, 1),
        'swap_xy': lambda p: p[:, [1, 0, 2]],
    }
    results = {}
    for name, transform in transforms.items():
        tested = transform(points.copy())
        distances, nearest = tree.query(tested[:, :2], k=1)
        vertical = np.abs(road[nearest, 2] - tested[:, 2])
        results[name] = {
            'samples': len(tested),
            'medianHorizontalMeters': float(np.median(distances)),
            'p95HorizontalMeters': float(np.percentile(distances, 95)),
            'medianVerticalMeters': float(np.median(vertical)),
            'p95VerticalMeters': float(np.percentile(vertical, 95)),
        }
    winner = min(results, key=lambda key: results[key]['medianHorizontalMeters'])
    print(json.dumps({
        'roadModelsLoaded': model_count,
        'roadVertices': len(road),
        'winner': winner,
        'results': results,
    }, indent=2))


if __name__ == '__main__':
    main()
