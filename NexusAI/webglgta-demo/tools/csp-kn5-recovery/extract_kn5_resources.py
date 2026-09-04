import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


def read_i32(stream):
    return struct.unpack('<i', stream.read(4))[0]


def read_string(stream):
    size = read_i32(stream)
    return stream.read(size).decode('utf-8', errors='replace')


def safe_name(name):
    leaf = name.replace('\\', '/').rsplit('/', 1)[-1]
    leaf = leaf.replace(chr(34), '_')
    return re.sub(r'[<>:/\\|?*\x00-\x1f]', '_', leaf) or 'unnamed_texture'


def extract_file(kn5_path, texture_dir, known_hashes):
    materials = []
    textures = []
    with kn5_path.open('rb') as stream:
        magic, version = struct.unpack('<6sI', stream.read(10))
        if magic != b'sc6969':
            raise ValueError('{}: unexpected KN5 magic {!r}'.format(kn5_path, magic))
        if version > 5:
            stream.read(4)

        for _ in range(read_i32(stream)):
            texture_type = read_i32(stream)
            original_name = read_string(stream)
            size = read_i32(stream)
            payload = stream.read(size)
            digest = hashlib.sha256(payload).hexdigest()
            stored_name = safe_name(original_name)
            existing = known_hashes.get(stored_name.lower())
            if existing and existing != digest:
                path = Path(stored_name)
                stored_name = '{}__{}__{}{}'.format(
                    path.stem, kn5_path.stem, digest[:8], path.suffix)
            known_hashes[stored_name.lower()] = digest
            destination = texture_dir / stored_name
            if not destination.exists():
                destination.write_bytes(payload)
            textures.append({
                'name': original_name,
                'storedName': stored_name,
                'type': texture_type,
                'size': size,
                'sha256': digest,
            })

        for material_index in range(read_i32(stream)):
            name = read_string(stream)
            shader = read_string(stream)
            unknown_short = struct.unpack('<h', stream.read(2))[0]
            unknown_int = read_i32(stream) if version > 4 else None
            properties = []
            for _ in range(read_i32(stream)):
                property_name = read_string(stream)
                value = struct.unpack('<f', stream.read(4))[0]
                extra = list(struct.unpack('<9f', stream.read(36)))
                properties.append({'name': property_name, 'value': value, 'extra': extra})
            bindings = []
            for _ in range(read_i32(stream)):
                slot = read_string(stream)
                slot_index = read_i32(stream)
                texture_name = read_string(stream)
                bindings.append({'slot': slot, 'slotIndex': slot_index, 'texture': texture_name})
            materials.append({
                'index': material_index,
                'name': name,
                'shader': shader,
                'unknownShort': unknown_short,
                'unknownInt': unknown_int,
                'properties': properties,
                'textures': bindings,
            })

    return {'file': kn5_path.name, 'version': version, 'textures': textures, 'materials': materials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('track_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    texture_dir = args.output_dir / 'textures'
    texture_dir.mkdir(exist_ok=True)
    known_hashes = {}
    catalog = []
    for kn5_path in sorted(args.track_dir.glob('*.kn5')):
        item = extract_file(kn5_path, texture_dir, known_hashes)
        catalog.append(item)
        print('{}: {} textures, {} materials'.format(
            kn5_path.name, len(item['textures']), len(item['materials'])))
    output = {
        'format': 'KN5_RESOURCE_CATALOG_1',
        'sources': catalog,
        'storedTextureCount': len(list(texture_dir.iterdir())),
    }
    (args.output_dir / 'materials.json').write_text(
        json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
