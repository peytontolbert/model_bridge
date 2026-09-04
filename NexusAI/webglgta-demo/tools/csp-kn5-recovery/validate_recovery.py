import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('recovery', type=Path)
    args = parser.parse_args()
    root = args.recovery
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    batches = {
        i: (root / f'batch_{i:05d}.bin').read_bytes()
        for i in range(manifest['batchCount'])
    }

    global_min = np.full(3, np.inf)
    global_max = np.full(3, -np.inf)
    edge_samples = []
    invalid_vertices = 0

    for mesh in manifest['meshes']:
        blob = batches[mesh['batch']]
        vo = mesh['vertexOffset']
        vb = mesh['vertexBytes']
        io = mesh['indexOffset']
        ib = mesh['indexBytes']
        vertices = np.frombuffer(blob, dtype='<f4', count=vb // 4, offset=vo).reshape(-1, 11)
        indices = np.frombuffer(blob, dtype='<u2', count=ib // 2, offset=io).astype(np.int64)
        positions = vertices[:, :3]
        finite = np.isfinite(positions).all(axis=1)
        invalid_vertices += int((~finite).sum())
        if finite.any():
            global_min = np.minimum(global_min, positions[finite].min(axis=0))
            global_max = np.maximum(global_max, positions[finite].max(axis=0))
        triangles = indices[: indices.size - indices.size % 3].reshape(-1, 3)
        valid_triangles = triangles[(triangles < len(positions)).all(axis=1)]
        if len(valid_triangles):
            stride = max(1, len(valid_triangles) // 100_000)
            tri_positions = positions[valid_triangles[::stride]]
            edge_samples.append(np.concatenate([
                np.linalg.norm(tri_positions[:, 0] - tri_positions[:, 1], axis=1),
                np.linalg.norm(tri_positions[:, 1] - tri_positions[:, 2], axis=1),
                np.linalg.norm(tri_positions[:, 2] - tri_positions[:, 0], axis=1),
            ]))

    edges = np.concatenate(edge_samples) if edge_samples else np.array([], dtype=np.float32)
    report = {
        'format': manifest['format'],
        'meshes': manifest['meshCount'],
        'vertices': manifest['vertexCount'],
        'indices': manifest['indexCount'],
        'invalidVertices': invalid_vertices,
        'boundsMin': global_min.tolist(),
        'boundsMax': global_max.tolist(),
        'edgeP50': float(np.percentile(edges, 50)) if len(edges) else None,
        'edgeP95': float(np.percentile(edges, 95)) if len(edges) else None,
        'edgeP99': float(np.percentile(edges, 99)) if len(edges) else None,
        'edgeMax': float(edges.max()) if len(edges) else None,
        'edgesOver100m': int((edges > 100).sum()),
        'sampledEdges': int(len(edges)),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
