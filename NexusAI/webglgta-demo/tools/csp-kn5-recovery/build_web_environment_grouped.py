import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


PIT_X = -43750.859375
PIT_Z = -19906.48046875
COMPONENTS = ('main', 'wall', 'lamp', 'bush', 'trees')


def category_for(component: str, material: str, name: str) -> str:
    key = f'{material} {name}'.lower()
    if 'emissive' in key or 'light lamp' in key or 'shell_logo' in key or 'screen' in key:
        return 'emissive'
    if 'glass' in key:
        return 'glass'
    if 'ligne jaune' in key or 'yellow' in key:
        return 'yellow'
    if 'ligne white' in key or 'white' in key:
        return 'white'
    if 'red' in key:
        return 'red'
    if 'tree' in key or component in ('bush', 'trees'):
        return 'vegetation'
    if 'metal' in key or 'chrome' in key or 'antenne' in key or component == 'lamp':
        return 'metal'
    if component == 'wall' or 'beton' in key or 'concrete' in key or 'barrier' in key:
        return 'wall'
    return 'main'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('recovery_root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--radius', type=float, default=2500.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    source_stats = {}

    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        temp = Path(temporary)
        for component in COMPONENTS:
            source = args.recovery_root / component
            manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
            source_triangles = 0
            selected_triangles = 0
            for mesh in manifest['meshes']:
                if mesh.get('physicsOnly'):
                    continue
                batch = source / f'batch_{mesh["batch"]:05d}.bin'
                vertices = np.memmap(batch, dtype='<f4', mode='r', offset=mesh['vertexOffset'], shape=(mesh['vertexCount'], 11))
                indices = np.memmap(batch, dtype='<u2', mode='r', offset=mesh['indexOffset'], shape=(mesh['indexCount'],)).astype(np.int64)
                triangles = indices[:indices.size - indices.size % 3].reshape(-1, 3)
                triangles = triangles[(triangles < len(vertices)).all(axis=1)]
                source_triangles += len(triangles)
                if not len(triangles):
                    continue
                centers = vertices[triangles, :3].mean(axis=1)
                inside = (centers[:, 0] - PIT_X) ** 2 + (centers[:, 2] - PIT_Z) ** 2 <= args.radius ** 2
                triangles = triangles[inside]
                if not len(triangles):
                    continue
                category = category_for(component, mesh.get('material', ''), mesh.get('name', ''))
                expanded = np.ascontiguousarray(vertices[triangles.reshape(-1), :8], dtype='<f4')
                with (temp / f'{category}.bin').open('ab') as destination:
                    destination.write(expanded.tobytes())
                counts[category] += len(expanded)
                selected_triangles += len(expanded) // 3
            source_stats[component] = {'sourceTriangles': source_triangles, 'selectedTriangles': selected_triangles}
            print(f'{component}: {selected_triangles:,} triangles', flush=True)

        first_vertex = 0
        segments = []
        with (args.output / 'geometry.bin').open('wb') as destination:
            for category in sorted(counts):
                with (temp / f'{category}.bin').open('rb') as source_file:
                    shutil.copyfileobj(source_file, destination, length=8 * 1024 * 1024)
                count = counts[category]
                segments.append({'component': category, 'material': category, 'name': category, 'firstVertex': first_vertex, 'vertexCount': count, 'triangleCount': count // 3})
                first_vertex += count

    result = {
        'format': 'CSP_WEB_ENVIRONMENT_MATERIAL_1',
        'vertexStride': 32,
        'radius': args.radius,
        'center': [PIT_X, PIT_Z],
        'vertexCount': first_vertex,
        'triangleCount': first_vertex // 3,
        'geometry': 'geometry.bin',
        'segments': segments,
        'sourceStats': source_stats,
    }
    (args.output / 'manifest.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
