import argparse
import json
import math
import re
import struct
from pathlib import Path


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, got {len(data)}")
    return data


def read_i32(stream):
    return struct.unpack("<i", read_exact(stream, 4))[0]


def read_u32(stream):
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_string(stream):
    length = read_i32(stream)
    if length < 0 or length > 16 * 1024 * 1024:
        raise ValueError(f"invalid KN5 string length {length} at 0x{stream.tell() - 4:x}")
    return read_exact(stream, length).decode("utf-8", errors="replace")


def mat_mul(a, b):
    return [sum(a[row * 4 + k] * b[k * 4 + col] for k in range(4))
            for row in range(4) for col in range(4)]


IDENTITY = [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


def skip_resources(stream, version):
    texture_count = read_i32(stream)
    for _ in range(texture_count):
        read_exact(stream, 4)
        read_string(stream)
        stream.seek(read_u32(stream), 1)

    material_count = read_i32(stream)
    for _ in range(material_count):
        read_string(stream)
        read_string(stream)
        read_exact(stream, 2)
        read_exact(stream, 4)  # depth mode is present in every KN5 version used by AC
        property_count = read_i32(stream)
        for _ in range(property_count):
            read_string(stream)
            read_exact(stream, 40)
        mapping_count = read_i32(stream)
        for _ in range(mapping_count):
            read_string(stream)
            read_exact(stream, 4)
            read_string(stream)

    return texture_count, material_count


def read_node(stream, parent_world, path, matches, pattern, stats):
    node_class = read_i32(stream)
    if node_class not in (1, 2, 3):
        raise ValueError(f"invalid node class {node_class} at 0x{stream.tell() - 4:x}")
    name = read_string(stream)
    child_count = read_i32(stream)
    active = bool(read_exact(stream, 1)[0])
    local = IDENTITY
    stats["nodeCount"] += 1

    if node_class == 1:
        local = list(struct.unpack("<16f", read_exact(stream, 64)))
    elif node_class == 2:
        read_exact(stream, 3)
        stream.seek(read_u32(stream) * 44, 1)
        stream.seek(read_u32(stream) * 2, 1)
        read_exact(stream, 33)
    else:
        read_exact(stream, 3)
        bone_count = read_u32(stream)
        for _ in range(bone_count):
            read_string(stream)
            read_exact(stream, 64)
        stream.seek(read_u32(stream) * 76, 1)
        stream.seek(read_u32(stream) * 2, 1)
        read_exact(stream, 16)

    # KN5/System.Numerics matrices use row-vector composition and M41/M42/M43 translation.
    world = mat_mul(local, parent_world)
    node_path = path + [name]
    if pattern.search(name):
        matches.append({
            "name": name,
            "path": "/".join(node_path),
            "nodeClass": node_class,
            "active": active,
            "position": [world[12], world[13], world[14]],
            "forward": [world[8], world[9], world[10]],
            "up": [world[4], world[5], world[6]],
            "worldTransform": world,
        })

    for _ in range(child_count):
        read_node(stream, world, node_path, matches, pattern, stats)


def extract(source, expression):
    matches = []
    stats = {"nodeCount": 0}
    pattern = re.compile(expression, re.IGNORECASE)
    with source.open("rb") as stream:
        magic = read_exact(stream, 6)
        if magic != b"sc6969":
            raise ValueError(f"not a KN5 file: {source}")
        version = read_i32(stream)
        extra = read_i32(stream) if version > 5 else None
        textures, materials = skip_resources(stream, version)
        hierarchy_offset = stream.tell()
        read_node(stream, IDENTITY, [], matches, pattern, stats)
        trailing_bytes = source.stat().st_size - stream.tell()
    return {
        "schema": "kn5-node-transforms/v1",
        "source": str(source),
        "version": version,
        "headerExtra": extra,
        "textureCount": textures,
        "materialCount": materials,
        "hierarchyOffset": hierarchy_offset,
        "nodeCount": stats["nodeCount"],
        "matchExpression": expression,
        "matchCount": len(matches),
        "trailingBytes": trailing_bytes,
        "instances": matches,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract named transforms from an Assetto Corsa KN5 hierarchy")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--match", default=r"^(AC_PIT|AC_START|AC_HOTLAP_START|AC_TIME)_?", help="case-insensitive node-name regex")
    args = parser.parse_args()
    result = extract(args.source.resolve(), args.match)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("nodeCount", "matchCount", "trailingBytes")}, indent=2))


if __name__ == "__main__":
    main()
