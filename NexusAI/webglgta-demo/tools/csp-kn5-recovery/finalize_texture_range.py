#!/usr/bin/env python3
"""Precompile a disjoint CSP texture ID range into a finalizer resume cache."""

import argparse
import concurrent.futures
import json
from pathlib import Path

from finalize_resolved_textures import load_manifest
from finalize_resolved_textures_corrected import compile_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--start-id', type=int, required=True)
    parser.add_argument('--end-id', type=int, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-web-dimension', type=int, default=1024)
    args = parser.parse_args()
    run = args.run.resolve()
    cache = args.cache.resolve()
    (cache / 'textures').mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(run, 'nohesi_110')
    records = [record for record in manifest['textures'] if args.start_id <= int(record['id']) <= args.end_id]
    empty_cache = cache.with_name(cache.name + '.none')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            compile_record, run, cache, empty_cache, record, args.max_web_dimension
        ) for record in records]
        results = [future.result() for future in futures]
    print(json.dumps({'converted': len(results), 'startId': args.start_id, 'endId': args.end_id}, indent=2))


if __name__ == '__main__':
    main()
