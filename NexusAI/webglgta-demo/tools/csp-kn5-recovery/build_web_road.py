import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.source / 'manifest.json').read_text(encoding='utf-8'))
    batches = {
        batch_id: (args.source / f'batch_{batch_id:05d}.bin').read_bytes()
        for batch_id in range(manifest['batchCount'])
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_meshes = []
    first_vertex = 0

    with (args.output / 'geometry.bin').open('wb') as destination:
        for mesh in manifest['meshes']:
            payload = batches[mesh['batch']]
            vertices = np.frombuffer(
                payload,
                dtype='<f4',
                count=mesh['vertexBytes'] // 4,
                offset=mesh['vertexOffset'],
            ).reshape(-1, 11)
            indices = np.frombuffer(
                payload,
                dtype='<u2',
                count=mesh['indexBytes'] // 2,
                offset=mesh['indexOffset'],
            ).astype(np.int64)
            triangles = indices[:indices.size - indices.size % 3].reshape(-1, 3)
            triangles = triangles[(triangles < len(vertices)).all(axis=1)]
            expanded = np.ascontiguousarray(vertices[triangles.reshape(-1), :8], dtype='<f4')
            destination.write(expanded.tobytes())
            count = len(expanded)
            output_meshes.append({
                'name': mesh['name'],
                'material': mesh['material'],
                'firstVertex': first_vertex,
                'vertexCount': count,
                'triangleCount': count // 3,
            })
            first_vertex += count

    web_manifest = {
        'format': 'CSP_WEB_ROAD_MATERIAL_1',
        'sourceFormat': manifest['format'],
        'vertexStride': 32,
        'vertexCount': first_vertex,
        'triangleCount': first_vertex // 3,
        'meshCount': len(output_meshes),
        'geometry': 'geometry.bin',
        'meshes': output_meshes,
    }
    (args.output / 'manifest.json').write_text(
        json.dumps(web_manifest, separators=(',', ':')),
        encoding='utf-8',
    )
    print(json.dumps(web_manifest | {'meshes': f'{len(output_meshes)} entries'}, indent=2))


if __name__ == '__main__':
    main()
