#!/usr/bin/env python3
"""Rewrite anticipated CSP texture references to final full-SHA WebP names."""

import argparse
import hashlib
import json
import re
from pathlib import Path


PATTERN = re.compile(r'^textures/(\d{4})_[0-9a-f]+\.webp$')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--recovery-manifest', type=Path, required=True)
    args = parser.parse_args()
    scene_path = args.scene.resolve()
    recovery_path = args.recovery_manifest.resolve()
    recovery = json.loads(recovery_path.read_text(encoding='utf-8'))
    names = {}
    for record in recovery['textures']:
        source = recovery_path.parent / record['outputFile']
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        names[int(record['id'])] = 'textures/{:04d}_{}.webp'.format(int(record['id']), digest)
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    changed = 0
    refs = set()
    for model in scene.get('models', []):
        for group in model.get('groups') or []:
            for channel, value in list((group.get('textures') or {}).items()):
                match = PATTERN.match(str(value))
                if not match:
                    continue
                replacement = names[int(match.group(1))]
                refs.add(replacement)
                if value != replacement:
                    group['textures'][channel] = replacement
                    changed += 1
    temporary = scene_path.with_suffix('.json.staging')
    temporary.write_text(json.dumps(scene, separators=(',', ':')), encoding='utf-8')
    temporary.replace(scene_path)
    print(json.dumps({'changedBindings': changed, 'distinctTextureReferences': len(refs)}, indent=2))


if __name__ == '__main__':
    main()
