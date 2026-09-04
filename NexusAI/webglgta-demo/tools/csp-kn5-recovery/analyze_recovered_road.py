import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
pit = (-43750.859375, 586.9697875976562, -19906.48046875)
best = (float('inf'), None, None)
files = {}
count = 0
all_local = []

for mesh in manifest['meshes']:
    batch_id = mesh['batch']
    if batch_id not in files:
        files[batch_id] = (root / f'batch_{batch_id:05d}.bin').read_bytes()
    payload = files[batch_id]
    vertices = np.frombuffer(
        payload,
        dtype='<f4',
        count=mesh['vertexBytes'] // 4,
        offset=mesh['vertexOffset'],
    ).reshape(-1, 11)[:, :3]
    all_local.append(np.column_stack((
        vertices[:, 0] - pit[0],
        vertices[:, 2] - pit[2],
        vertices[:, 1] - pit[1],
    )))
    for index in range(mesh['vertexCount']):
        offset = mesh['vertexOffset'] + index * manifest['vertexStride']
        point = struct.unpack_from('<3f', payload, offset)
        distance_sq = sum((point[axis] - pit[axis]) ** 2 for axis in range(3))
        count += 1
        if distance_sq < best[0]:
            best = (distance_sq, point, mesh['name'])

point = best[1]
local = np.concatenate(all_local)
focus = np.array([-0.16015625, -30.990234375, -0.09442138671875])
yaw = -1.47
pitch = 0.24
distance = 190
cp = math.cos(pitch)
eye = focus + np.array([
    distance * math.cos(yaw) * cp,
    distance * math.sin(yaw) * cp,
    distance * math.sin(pitch) + 24,
])
forward = focus - eye
forward /= np.linalg.norm(forward)
right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
right /= np.linalg.norm(right)
up = np.cross(right, forward)
relative = local - eye
depth = relative @ forward
screen_x = relative @ right
screen_y = relative @ up
tan_y = math.tan(math.pi / 6)
visible = (depth > 0.5) & (depth < 80000) & (np.abs(screen_y) < depth * tan_y) & (np.abs(screen_x) < depth * tan_y * (1120 / 508))
near = local[np.linalg.norm(local[:, :2] - focus[:2], axis=1) < 400]
covariance = np.cov(near[:, :2], rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
road_angle = math.atan2(tangent[1], tangent[0])
print(json.dumps({
    'vertices': count,
    'nearestDistance': math.sqrt(best[0]),
    'nearestXYZ': point,
    'nearestLocalZup': [point[0] - pit[0], point[2] - pit[2], point[1] - pit[1]],
    'mesh': best[2],
    'cameraEye': eye.tolist(),
    'verticesInFrustum': int(visible.sum()),
    'nearbyVertices': len(near),
    'roadTangentYaw': road_angle,
}, indent=2))
