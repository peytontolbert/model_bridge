import argparse
import json
from pathlib import Path

import numpy as np


PIT_X = -43750.859375
PIT_Z = -19906.48046875
COMPONENTS = ('main', 'wall', 'lamp', 'bush', 'trees')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('recovery_root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--radius', type=float, default=2500.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / 'geometry.bin'
    segments = []
    first_vertex = 0

    with output_path.open('wb') as destination:
        for component in COMPONENTS:
            source = args.recovery_root / component
            manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
            batches = {
                batch_id: source / f'batch_{batch_id:05d}.bin'
                for batch_id in range(manifest['batchCount'])
            }
            source_triangles = 0
            selected_triangles = 0
            material_groups = {}

            for mesh in manifest['meshes']:
                if mesh.get('physicsOnly'):
                    continue
                vertex_count = mesh['vertexCount']
                vertices = np.memmap(
                    batches[mesh['batch']],
                    dtype='<f4',
                    mode='r',
                    offset=mesh['vertexOffset'],
                    shape=(vertex_count, 11),
                )
                indices = np.memmap(
                    batches[mesh['batch']],
                    dtype='<u2',
                    mode='r',
                    offset=mesh['indexOffset'],
                    shape=(mesh['indexCount'],),
                ).astype(np.int64)
                triangles = indices[:indices.size - indices.size % 3].reshape(-1, 3)
                triangles = triangles[(triangles < vertex_count).all(axis=1)]
                source_triangles += len(triangles)
                if not len(triangles):
                    continue
                centers = vertices[triangles, :3].mean(axis=1)
                distance_sq = (centers[:, 0] - PIT_X) ** 2 + (centers[:, 2] - PIT_Z) ** 2
                triangles = triangles[distance_sq <= args.radius ** 2]
                if not len(triangles):
                    continue
                expanded = np.ascontiguousarray(vertices[triangles.reshape(-1), :8], dtype='<f4')
                added = len(expanded)
                selected_triangles += added // 3
                material = mesh.get('material', '')
                group = material_groups.setdefault(material, {
                    'name': mesh.get('name', ''),
                    'parts': [],
                    'vertexCount': 0,
                })
                group['parts'].append(expanded.tobytes())
                group['vertexCount'] += added

            for material, group in material_groups.items():
                group_first = first_vertex
                destination.write(b''.join(group['parts']))
                first_vertex += group['vertexCount']
                segments.append({
                    'component': component,
                    'material': material,
                    'name': group['name'],
                    'firstVertex': group_first,
                    'vertexCount': group['vertexCount'],
                    'triangleCount': group['vertexCount'] // 3,
                })
            print(f'{component}: {selected_triangles:,} of {source_triangles:,} triangles', flush=True)

    web_manifest = {
        'format': 'CSP_WEB_ENVIRONMENT_MATERIAL_1',
        'vertexStride': 32,
        'radius': args.radius,
        'center': [PIT_X, PIT_Z],
        'vertexCount': first_vertex,
        'triangleCount': first_vertex // 3,
        'geometry': output_path.name,
        'segments': segments,
    }
    (args.output / 'manifest.json').write_text(
        json.dumps(web_manifest, separators=(',', ':')),
        encoding='utf-8',
    )
    print(json.dumps(web_manifest, indent=2))


if __name__ == '__main__':
    main()
