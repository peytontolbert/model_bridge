#!/usr/bin/env python3
"""Build a browser guide overlay from Assetto traffic data and CSP blocker meshes."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


BLOCKER = re.compile(r'blockade|prop[_ ]?blockers?', re.I)
IGNORE = re.compile(r'light[_ ]?blocker', re.I)


def transform_positions(vertices: np.ndarray, transform: dict) -> np.ndarray:
    side = np.asarray(transform.get('side', [1, 0, 0]), dtype=np.float32)
    up = np.asarray(transform.get('up', [0, 1, 0]), dtype=np.float32)
    look = np.asarray(transform.get('look', [0, 0, 1]), dtype=np.float32)
    position = np.asarray(transform.get('position', [0, 0, 0]), dtype=np.float32)
    source = vertices[:, :3] @ np.stack((side, up, look), axis=1).T + position
    return source[:, [0, 2, 1]]


def blocker_records(component_root: Path) -> list[dict]:
    records = []
    for manifest_path in sorted(component_root.glob('*/manifest.json')):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        source = str(manifest.get('sourceFile') or manifest_path.parent.name)
        for mesh in manifest.get('meshes') or []:
            # Match physical closure nodes, not emissive materials such as
            # M_CT_Prop_blockers_DA used by a large decorative light mesh.
            identity = str(mesh.get('name') or '')
            if not BLOCKER.search(identity) or IGNORE.search(identity):
                continue
            batch_path = manifest_path.parent / f"batch_{int(mesh['batch']):05d}.bin"
            vertices = np.memmap(
                batch_path, dtype='<f4', mode='r', offset=int(mesh['vertexOffset']),
                shape=(int(mesh['vertexCount']), 11),
            )
            positions = transform_positions(vertices, mesh.get('worldTransform') or {})
            minimum = positions.min(axis=0)
            maximum = positions.max(axis=0)
            center = (minimum + maximum) * 0.5
            records.append({
                'id': f"blocker-{len(records) + 1:03d}",
                'name': str(mesh.get('name') or ''),
                'source': source,
                'material': str(mesh.get('material') or ''),
                'center': center.round(6).tolist(),
                'bounds': {'min': minimum.round(6).tolist(), 'max': maximum.round(6).tolist()},
            })
    return records


def blocked_zones(blockers: list[dict], cluster_distance: float) -> list[dict]:
    if not blockers:
        return []
    centers = np.asarray([record['center'][:2] for record in blockers], dtype=np.float64)
    parent = list(range(len(blockers)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(blockers)):
        distances = np.linalg.norm(centers[left + 1:] - centers[left], axis=1)
        for offset in np.flatnonzero(distances <= cluster_distance):
            union(left, left + 1 + int(offset))
    groups = {}
    for index in range(len(blockers)):
        groups.setdefault(find(index), []).append(index)
    zones = []
    for indices in groups.values():
        minimum = np.min([blockers[index]['bounds']['min'] for index in indices], axis=0)
        maximum = np.max([blockers[index]['bounds']['max'] for index in indices], axis=0)
        center = (minimum + maximum) * 0.5
        zones.append({
            'id': f"blocked-zone-{len(zones) + 1:02d}",
            'label': f"Authored road closure {len(zones) + 1}",
            'center': center.round(6).tolist(),
            'bounds': {'min': minimum.round(6).tolist(), 'max': maximum.round(6).tolist()},
            'blockerIds': [blockers[index]['id'] for index in indices],
            'meshCount': len(indices),
        })
    return sorted(zones, key=lambda zone: (zone['center'][0], zone['center'][1]))


def cutoff_records(traffic_zones: list[dict], blockers: list[dict], threshold: float) -> list[dict]:
    if not blockers:
        return []
    blocker_xy = np.asarray([record['center'][:2] for record in blockers], dtype=np.float64)
    cutoffs = []
    for zone in traffic_zones:
        centerline = zone.get('centerline') or []
        if not centerline:
            continue
        for end_name, point in (('entry', centerline[0]), ('exit', centerline[-1])):
            xy = np.asarray(point[:2], dtype=np.float64)
            distances = np.linalg.norm(blocker_xy - xy, axis=1)
            blocker_index = int(np.argmin(distances))
            distance = float(distances[blocker_index])
            if distance > threshold:
                continue
            cutoffs.append({
                'id': f"cutoff-lane-{zone['laneId']}-{end_name}",
                'laneId': zone['laneId'],
                'end': end_name,
                'position': point,
                'blockerId': blockers[blocker_index]['id'],
                'blockerDistanceMeters': round(distance, 3),
            })
    return cutoffs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gameplay-zones', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff-distance', type=float, default=75.0)
    parser.add_argument('--blocker-cluster-distance', type=float, default=80.0)
    args = parser.parse_args()

    gameplay = json.loads(args.gameplay_zones.read_text(encoding='utf-8'))
    zones = gameplay.get('trafficZones') or []
    blockers = blocker_records(args.components)
    closures = blocked_zones(blockers, args.blocker_cluster_distance)
    cutoffs = cutoff_records(zones, blockers, args.cutoff_distance)
    result = {
        'schema': 'nohesi-110-viewer-guide/v1',
        'coordinateSystem': gameplay.get('coordinateSystem'),
        'source': {
            'traffic': str(args.gameplay_zones),
            'blockers': 'CSP loader mesh names matching Blockade/Prop_blockers',
            'cutoffRule': f"lane endpoint within {args.cutoff_distance:g} m of blocker center",
        },
        'trafficZones': zones,
        'blockers': blockers,
        'blockedZones': closures,
        'cutoffs': cutoffs,
        'stats': {
            'trafficZones': len(zones),
            'trafficCenterlinePoints': sum(len(zone.get('centerline') or []) for zone in zones),
            'blockerMeshes': len(blockers),
            'blockedZones': len(closures),
            'derivedCutoffs': len(cutoffs),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    print(json.dumps(result['stats'], indent=2))


if __name__ == '__main__':
    main()
