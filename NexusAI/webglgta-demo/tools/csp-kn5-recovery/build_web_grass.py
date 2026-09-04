import argparse
import json
from pathlib import Path

import numpy as np


PIT_X = -43750.859375
PIT_Z = -19906.48046875


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--radius', type=float, default=2500.0)
    parser.add_argument('--blades', type=int, default=40000)
    args = parser.parse_args()

    manifest = json.loads((args.source / 'manifest.json').read_text(encoding='utf-8'))
    positions = []
    weights = []
    for mesh in manifest['meshes']:
        batch = args.source / f'batch_{mesh["batch"]:05d}.bin'
        vertices = np.memmap(batch, dtype='<f4', mode='r', offset=mesh['vertexOffset'], shape=(mesh['vertexCount'], 11))
        indices = np.memmap(batch, dtype='<u2', mode='r', offset=mesh['indexOffset'], shape=(mesh['indexCount'],)).astype(np.int64)
        triangles = indices[:indices.size - indices.size % 3].reshape(-1, 3)
        triangles = triangles[(triangles < len(vertices)).all(axis=1)]
        triangle_positions = np.asarray(vertices[triangles, :3])
        centers = triangle_positions.mean(axis=1)
        inside = (centers[:, 0] - PIT_X) ** 2 + (centers[:, 2] - PIT_Z) ** 2 <= args.radius ** 2
        triangle_positions = triangle_positions[inside]
        area = np.linalg.norm(np.cross(triangle_positions[:, 1] - triangle_positions[:, 0], triangle_positions[:, 2] - triangle_positions[:, 0]), axis=1) * 0.5
        valid = np.isfinite(area) & (area > 0.01)
        positions.append(triangle_positions[valid])
        weights.append(area[valid])

    triangles = np.concatenate(positions)
    area = np.concatenate(weights)
    rng = np.random.default_rng(415)
    chosen = rng.choice(len(triangles), size=args.blades, replace=True, p=area / area.sum())
    selected = triangles[chosen]
    r1 = np.sqrt(rng.random(args.blades))
    r2 = rng.random(args.blades)
    roots = (1 - r1)[:, None] * selected[:, 0] + (r1 * (1 - r2))[:, None] * selected[:, 1] + (r1 * r2)[:, None] * selected[:, 2]
    heights = rng.uniform(2.0, 4.25, args.blades)
    widths = rng.uniform(0.22, 0.55, args.blades)
    angles = rng.uniform(0, np.pi, args.blades)
    cells = np.array([[1, 1], [2, 1], [6, 1], [7, 1], [3, 1], [6, 2], [7, 2]])
    chances = np.array([0.20, 0.35, 0.17, 0.17, 0.06, 0.03, 0.02])
    atlas = cells[rng.choice(len(cells), size=args.blades, p=chances / chances.sum())]
    output = np.empty((args.blades, 12, 5), dtype='<f4')

    for plane, rotation in enumerate((0.0, np.pi * 0.5)):
        direction_x = np.cos(angles + rotation) * widths
        direction_z = np.sin(angles + rotation) * widths
        left = roots.copy()
        right = roots.copy()
        left[:, 0] -= direction_x
        left[:, 2] -= direction_z
        right[:, 0] += direction_x
        right[:, 2] += direction_z
        left_top = left.copy()
        right_top = right.copy()
        left_top[:, 1] += heights
        right_top[:, 1] += heights
        base = plane * 6
        output[:, base:base + 6, :3] = np.stack((left, right, left_top, left_top, right, right_top), axis=1)
        u0 = atlas[:, 0] / 8.0
        u1 = (atlas[:, 0] + 1) / 8.0
        v0 = atlas[:, 1] / 3.0
        v1 = (atlas[:, 1] + 1) / 3.0
        output[:, base:base + 6, 3:] = np.stack((
            np.column_stack((u0, v1)), np.column_stack((u1, v1)), np.column_stack((u0, v0)),
            np.column_stack((u0, v0)), np.column_stack((u1, v1)), np.column_stack((u1, v0)),
        ), axis=1)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'grass.bin').write_bytes(output.tobytes())
    result = {
        'format': 'CSP_WEB_GRASS_1',
        'vertexStride': 20,
        'vertexCount': args.blades * 12,
        'triangleCount': args.blades * 4,
        'bladeCount': args.blades,
        'radius': args.radius,
        'geometry': 'grass.bin',
        'texture': 'highlands-darker.webp',
        'sourceSurfaceArea': float(area.sum()),
    }
    (args.output / 'manifest.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
