#!/usr/bin/env python3
"""Reuse an existing browser texture package by exact material/channel binding."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def source_key(value):
    return Path(str(value or '').replace('\\', '/').split('#', 1)[0]).name.lower()


def key(source, group):
    return source_key(source), str(group.get('material') or '').casefold(), str(group.get('shader') or '').casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-scene', type=Path, required=True)
    parser.add_argument('--target-scene', type=Path, required=True)
    args = parser.parse_args()
    old_path = args.existing_scene.resolve()
    new_path = args.target_scene.resolve()
    old = json.loads(old_path.read_text(encoding='utf-8'))
    new = json.loads(new_path.read_text(encoding='utf-8'))
    bindings = old.get('materialBindings', {})
    exact = {}
    fallback = {}
    for model in old.get('models', []):
        source = model.get('source') or model.get('tileOf') or model.get('file')
        for group in model.get('groups') or []:
            textures = dict((bindings.get(group.get('materialBinding')) or {}).get('textures') or {})
            textures.update(group.get('textures') or {})
            exact.setdefault(key(source, group), textures)
            fallback.setdefault(key(source, group)[:2], textures)
    copied = {}
    missing = set()
    conflicts = []
    for model in new.get('models', []):
        source = model.get('source') or model.get('tileOf') or model.get('file')
        for group in model.get('groups') or []:
            target_textures = group.get('textures') or {}
            source_textures = exact.get(key(source, group)) or fallback.get(key(source, group)[:2]) or {}
            for channel, target_ref in target_textures.items():
                source_ref = source_textures.get(channel)
                if not source_ref:
                    missing.add((source_key(source), group.get('material'), group.get('shader'), channel, target_ref))
                    continue
                source_file = old_path.parent / source_ref
                target_file = new_path.parent / target_ref
                if not source_file.is_file():
                    missing.add((source_key(source), group.get('material'), group.get('shader'), channel, target_ref))
                    continue
                source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
                previous = copied.get(str(target_file))
                if previous and previous != source_hash:
                    conflicts.append((str(target_file), previous, source_hash))
                    continue
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if not target_file.exists():
                    shutil.copy2(source_file, target_file)
                copied[str(target_file)] = source_hash
    report = {'copied': len(copied), 'missing': len(missing), 'conflicts': len(conflicts),
              'missingExamples': list(sorted(missing))[:20], 'conflictExamples': conflicts[:20]}
    print(json.dumps(report, indent=2))
    if missing or conflicts:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
