from __future__ import annotations

import argparse
import json
import sys

from .consolidation import load_consolidation_plan
from .integration_skeleton import IntegrationSkeletonError, plan_integration_skeleton, write_integration_skeleton
from .next_candidates import load_next_integration_plan, to_json as next_candidate_to_json
from .patches import load_patch_registry, validate_catalog_patches
from .runtime.causal_lm import CausalLMRequest, generate_causal_lm, to_json as causal_lm_to_json
from .runtime.classic_transformers import (
    ClassicTransformersRequest,
    inspect_classic_transformers,
    to_json as classic_transformers_to_json,
)
from .runtime.cosmos25 import Cosmos25Request, plan_or_run_cosmos25, to_json as cosmos25_to_json
from .registry import load_catalog
from .runtime.nemo_asr import NemoASRRequest, to_json as nemo_asr_to_json, transcribe_nemo_asr
from .runtime.workers import load_worker_registry, preflight_worker, to_json as worker_to_json
from .runtime.three_d_gen import (
    ThreeDGenBridgeError,
    ThreeDGenRequest,
    compare_trellis_hunyuan3d,
    generate_3d,
    list_3d_backends,
    preflight_3d_backend,
    to_json,
)


def _add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lane", help="Filter by bridge lane.")
    parser.add_argument("--status", help="Filter by status.")
    parser.add_argument(
        "--runnable",
        action="store_true",
        help="Show only entries with a validated runnable path.",
    )
    parser.add_argument("--env-policy", help="Filter by environment policy.")
    parser.add_argument("--consolidation-decision", help="Filter by consolidation decision.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="legacy-model-bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List cataloged bridge entries.")
    _add_common_filters(list_parser)

    doctor_parser = subparsers.add_parser("doctor", help="Show compatibility metadata for one model.")
    doctor_parser.add_argument("model_id")

    env_parser = subparsers.add_parser("env-matrix", help="Show model counts by preferred environment.")

    next_parser = subparsers.add_parser("next-candidates", help="List reviewed next legacy-model integration targets.")
    next_parser.add_argument("--lane")
    next_parser.add_argument("--env")
    next_parser.add_argument("--catalog-state")
    next_parser.add_argument("--limit", type=int)
    next_parser.add_argument("--json", action="store_true", help="Print full candidate records as JSON.")

    patches_parser = subparsers.add_parser("patches", help="List registered compatibility patches.")
    patches_parser.add_argument("--lane", help="Filter by bridge lane.")
    patches_parser.add_argument("--status", help="Filter by patch status.")

    patch_doctor_parser = subparsers.add_parser("patch-doctor", help="Validate catalog patch references.")

    gen_parser = subparsers.add_parser("generate-integration", help="Create an integration skeleton from the bridge catalog.")
    gen_parser.add_argument("model_id")
    gen_parser.add_argument("--catalog", default="data/bridge_catalog.json")
    gen_parser.add_argument("--out-dir", default="integrations")
    gen_parser.add_argument("--name")
    gen_parser.add_argument("--status", default="planned", choices=["planned", "experimental", "beta", "stable", "blocked"])
    gen_parser.add_argument("--force", action="store_true")
    gen_parser.add_argument("--dry-run", action="store_true")
    gen_parser.add_argument("--include", action="append", default=[])
    gen_parser.add_argument("--test-style", default="skip", choices=["skip", "xfail"])
    gen_parser.add_argument("--allow-uncataloged", action="store_true")
    gen_parser.add_argument("--lane")
    gen_parser.add_argument("--preferred-env")

    consolidate_parser = subparsers.add_parser("consolidation", help="Report environment consolidation decisions.")
    consolidate_parser.add_argument("--current-env", help="Filter by current env or env label.")
    consolidate_parser.add_argument("--decision", help="Filter by consolidation decision.")
    consolidate_parser.add_argument("--lane", help="Filter by bridge lane.")
    consolidate_parser.add_argument("--caller-python", help="Filter by supported caller Python major.minor, for example 3.12.")
    consolidate_parser.add_argument("--summary", action="store_true", help="Print decision and env counts.")

    three_d_parser = subparsers.add_parser("three-d", help="Inspect or run 3D generation bridge backends.")
    three_d_subparsers = three_d_parser.add_subparsers(dest="three_d_command", required=True)

    three_d_subparsers.add_parser("backends", help="List bridge-managed 3D backends.")
    three_d_subparsers.add_parser("conflicts", help="Show the TRELLIS.2 vs Hunyuan3D compatibility decision.")

    three_d_preflight = three_d_subparsers.add_parser("preflight", help="Check backend worker Python before model load.")
    three_d_preflight.add_argument("backend", choices=["hunyuan3d", "trellis"])
    three_d_preflight.add_argument("--timeout-sec", type=int, default=30)

    three_d_run = three_d_subparsers.add_parser("run", help="Dispatch a 3D generation request through the bridge worker.")
    three_d_run.add_argument("backend", choices=["hunyuan3d", "trellis"])
    three_d_run.add_argument("--image-path", required=True)
    three_d_run.add_argument("--output-dir", required=True)
    three_d_run.add_argument("--model-path")
    three_d_run.add_argument("--variant")
    three_d_run.add_argument("--output-format", default="glb")
    three_d_run.add_argument("--seed", type=int, default=42)
    three_d_run.add_argument("--texture", action="store_true")
    three_d_run.add_argument("--dry-run", action="store_true")
    three_d_run.add_argument("--timeout-sec", type=int)
    three_d_run.add_argument("--cuda-visible-devices", help="Set CUDA_VISIBLE_DEVICES for the backend worker.")
    three_d_run.add_argument("--extra-json", help="JSON object merged into request.extra_args.")

    workers_parser = subparsers.add_parser("workers", help="Inspect bridge-owned backend workers.")
    workers_subparsers = workers_parser.add_subparsers(dest="workers_command", required=True)

    workers_list = workers_subparsers.add_parser("list", help="List registered workers.")
    workers_list.add_argument("--lane")
    workers_list.add_argument("--env")
    workers_list.add_argument("--status")

    workers_doctor = workers_subparsers.add_parser("doctor", help="Show one worker spec.")
    workers_doctor.add_argument("worker_or_model")
    workers_doctor.add_argument("--model", action="store_true", help="Resolve the argument as a model id.")

    workers_preflight = workers_subparsers.add_parser("preflight", help="Run worker Python/import preflight.")
    workers_preflight.add_argument("worker_or_model")
    workers_preflight.add_argument("--model", action="store_true", help="Resolve the argument as a model id.")
    workers_preflight.add_argument("--timeout-sec", type=int, default=30)

    cosmos_parser = subparsers.add_parser("cosmos25", help="Plan or run Cosmos 2.5 official worker requests.")
    cosmos_subparsers = cosmos_parser.add_subparsers(dest="cosmos_command", required=True)
    cosmos_plan = cosmos_subparsers.add_parser("plan", help="Validate Cosmos 2.5 assets and render official launch plan.")
    cosmos_plan.add_argument("--model-id", required=True, choices=["nvidia/Cosmos-Predict2.5-14B", "nvidia/Cosmos-Transfer2.5-2B"])
    cosmos_plan.add_argument("--input", action="append", default=[], help="Official Cosmos input JSON. Can be repeated.")
    cosmos_plan.add_argument("--output-dir", default="/data/tmp/legacy_model_bridge_cosmos25")
    cosmos_plan.add_argument("--checkpoint-path")
    cosmos_plan.add_argument("--repo-root")
    cosmos_plan.add_argument("--model")
    cosmos_plan.add_argument("--inference-type")
    cosmos_plan.add_argument("--nproc-per-node", type=int, default=1)
    cosmos_plan.add_argument("--master-port", type=int, default=12341)
    cosmos_plan.add_argument("--context-parallel-size", type=int)
    cosmos_plan.add_argument("--cuda-visible-devices")
    cosmos_plan.add_argument("--allow-downloads", action="store_true", help="Do not force HF_HUB_OFFLINE=1 in the rendered launch plan.")
    cosmos_plan.add_argument("--enable-guardrails", action="store_true")
    cosmos_plan.add_argument("--offload-diffusion-model", action="store_true")
    cosmos_plan.add_argument("--offload-text-encoder", action="store_true")
    cosmos_plan.add_argument("--offload-tokenizer", action="store_true")
    cosmos_plan.add_argument("--student-only", action="store_true", help="Use the bridge DMD2 student-only patch for Cosmos Transfer 2.5.")
    cosmos_plan.add_argument("--dry-run", action="store_true")
    cosmos_plan.add_argument("--timeout-sec", type=int)
    cosmos_plan.add_argument("--extra-json", help="JSON object merged into official CLI args.")

    causal_lm_parser = subparsers.add_parser("causal-lm", help="Run Transformers causal LM bridge requests.")
    causal_lm_subparsers = causal_lm_parser.add_subparsers(dest="causal_lm_command", required=True)
    causal_lm_generate = causal_lm_subparsers.add_parser("generate", help="Generate text through the causal LM bridge.")
    causal_lm_generate.add_argument("--model-id", required=True)
    causal_lm_generate.add_argument("--prompt", default="Hello")
    causal_lm_generate.add_argument("--model-root", default="/arxiv/models")
    causal_lm_generate.add_argument("--max-new-tokens", type=int, default=4)
    causal_lm_generate.add_argument("--device", default="cuda:0")
    causal_lm_generate.add_argument("--dtype", default="auto")
    causal_lm_generate.add_argument("--trust-remote-code", action="store_true")
    causal_lm_generate.add_argument("--no-trust-remote-code", action="store_true")
    causal_lm_generate.add_argument("--use-cache", action="store_true")
    causal_lm_generate.add_argument("--no-use-cache", action="store_true")
    causal_lm_generate.add_argument("--dry-run", action="store_true")

    classic_parser = subparsers.add_parser("classic-transformers", help="Inspect or run classic Transformers bridge requests.")
    classic_subparsers = classic_parser.add_subparsers(dest="classic_command", required=True)
    classic_inspect = classic_subparsers.add_parser("inspect", help="Inspect or synthetic-smoke a classic Transformers model.")
    classic_inspect.add_argument("--model-id", required=True)
    classic_inspect.add_argument("--model-root", default="/arxiv/models")
    classic_inspect.add_argument("--task", choices=["seq2seq_generation", "audio_encoder", "vision_encoder"])
    classic_inspect.add_argument("--device", default="cpu")
    classic_inspect.add_argument("--dtype", default="auto")
    classic_inspect.add_argument("--prompt", default="Legacy model bridge smoke input.")
    classic_inspect.add_argument("--max-new-tokens", type=int, default=4)
    classic_inspect.add_argument("--run-synthetic", action="store_true")
    classic_inspect.add_argument("--trust-remote-code", action="store_true")

    nemo_parser = subparsers.add_parser("nemo-asr", help="Run NeMo ASR archive bridge requests.")
    nemo_subparsers = nemo_parser.add_subparsers(dest="nemo_command", required=True)
    nemo_run = nemo_subparsers.add_parser("run", help="Restore or transcribe through the NeMo ASR worker.")
    nemo_run.add_argument("--model-id", default="parakeet-tdt_ctc-110m")
    nemo_run.add_argument("--archive-path")
    nemo_run.add_argument("--audio", action="append", default=[], help="Audio file to transcribe. Can be repeated.")
    nemo_run.add_argument("--output-dir", default="/data/tmp/legacy_model_bridge_nemo_asr")
    nemo_run.add_argument("--map-location", default="cpu")
    nemo_run.add_argument("--device", default="cuda:0")
    nemo_run.add_argument("--inspect-only", action="store_true", help="Inspect archive/env/target without restoring.")
    nemo_run.add_argument("--load-only", action="store_true", help="Restore and move the model without transcribing.")
    nemo_run.add_argument("--force-restore", action="store_true")
    nemo_run.add_argument("--dry-run", action="store_true")
    nemo_run.add_argument("--timeout-sec", type=int)

    args = parser.parse_args(argv)
    catalog = load_catalog()

    if args.command == "list":
        entries = catalog.filter(
            lane=args.lane,
            status=args.status,
            runnable=True if args.runnable else None,
            env_policy=args.env_policy,
            consolidation_decision=args.consolidation_decision,
        )
        for entry in entries:
            runnable = "yes" if entry.runnable else "no"
            print(
                f"{entry.model_id}\t{entry.lane}\t{entry.status}\t"
                f"{entry.preferred_env}->{entry.target_env}\t{entry.env_policy}\t"
                f"{entry.consolidation_decision}\trunnable={runnable}"
            )
        return 0

    if args.command == "doctor":
        try:
            entry = catalog.get(args.model_id)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"model_id: {entry.model_id}")
        print(f"lane: {entry.lane}")
        print(f"status: {entry.status}")
        print(f"preferred_env: {entry.preferred_env}")
        print(f"target_env: {entry.target_env}")
        print(f"env_policy: {entry.env_policy}")
        print(f"worker_boundary: {entry.worker_boundary}")
        print(f"worker_entrypoint: {entry.worker_entrypoint or 'none'}")
        print(f"artifact_contract: {entry.artifact_contract or 'none'}")
        print(f"consolidation_decision: {entry.consolidation_decision}")
        print(f"consolidation_blockers: {', '.join(entry.consolidation_blockers) or 'none'}")
        print(f"caller_python: {', '.join(entry.caller_python) or 'unknown'}")
        print(f"backend_python: {', '.join(entry.backend_python) or 'unknown'}")
        print(f"python_strategy: {entry.python_strategy}")
        print(f"mismatch_classes: {', '.join(entry.mismatch_classes) or 'none'}")
        print(f"runnable: {str(entry.runnable).lower()}")
        print(f"compatibility_patches: {', '.join(entry.compatibility_patches) or 'none'}")
        print(f"source_refs: {', '.join(entry.source_refs) or 'none'}")
        print(f"notes: {entry.notes}")
        return 0

    if args.command == "env-matrix":
        for env, entries in sorted(catalog.env_matrix().items()):
            runnable = sum(1 for entry in entries if entry.runnable)
            print(f"{env}\tmodels={len(entries)}\trunnable={runnable}")
        return 0

    if args.command == "next-candidates":
        plan = load_next_integration_plan()
        candidates = plan.filter(
            lane=args.lane,
            env=args.env,
            catalog_state=args.catalog_state,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps([next_candidate_to_json(candidate) for candidate in candidates], indent=2, sort_keys=True))
            return 0
        for candidate in candidates:
            mismatches = ",".join(candidate.mismatch_classes) or "none"
            print(
                f"{candidate.rank}\t{candidate.model_id}\t{candidate.lane}\t"
                f"env={candidate.preferred_env}\tstate={candidate.catalog_state}\t"
                f"mismatches={mismatches}"
            )
        return 0

    if args.command == "patches":
        registry = load_patch_registry()
        for patch in registry.filter(lane=args.lane, status=args.status):
            print(f"{patch.patch_id}\t{patch.lane}\t{patch.status}\t{patch.model_family}")
        return 0

    if args.command == "patch-doctor":
        missing = validate_catalog_patches()
        if not missing:
            print("all catalog patch references are registered")
            return 0
        for model_id, patch_ids in missing.items():
            print(f"{model_id}: missing {', '.join(patch_ids)}", file=sys.stderr)
        return 3

    if args.command == "consolidation":
        plan = load_consolidation_plan()
        if args.summary:
            print(f"latest_env: {plan.metadata.get('latest_env', 'unknown')}")
            for decision, count in sorted(plan.decision_counts().items()):
                print(f"decision\t{decision}\t{count}")
            for env, count in sorted(plan.current_env_counts().items()):
                print(f"current_env\t{env}\t{count}")
            for version, entries in sorted(plan.caller_python_matrix().items()):
                print(f"caller_python\t{version}\t{len(entries)}")
            return 0
        for entry in plan.filter(
            current_env=args.current_env,
            decision=args.decision,
            lane=args.lane,
            caller_python=args.caller_python,
        ):
            patches = ",".join(entry.required_patches) or "none"
            backend_python = ",".join(entry.backend_python) or "unknown"
            caller_python = ",".join(entry.caller_python) or "unknown"
            mismatches = ",".join(entry.mismatch_classes) or "none"
            print(
                f"{entry.model_id}\t{entry.current_env}->{entry.target_env}\t"
                f"{entry.decision}\tcaller_python={caller_python}\t"
                f"backend_python={backend_python}\tmismatches={mismatches}\t"
                f"patches={patches}\tblocker={entry.blocker}"
            )
        return 0

    if args.command == "three-d":
        if args.three_d_command == "backends":
            for backend in list_3d_backends():
                print(
                    f"{backend.backend}\t{backend.model_id}\tuser_env={backend.user_env}\t"
                    f"worker_env={backend.env}\tboundary={backend.worker_boundary}\t"
                    f"artifact_contract={backend.artifact_contract}"
                )
            return 0
        if args.three_d_command == "conflicts":
            report = compare_trellis_hunyuan3d()
            print(json.dumps(to_json(report), indent=2, sort_keys=True))
            return 0
        if args.three_d_command == "preflight":
            result = preflight_3d_backend(args.backend, timeout_sec=args.timeout_sec)
            print(json.dumps(to_json(result), indent=2, sort_keys=True))
            return 0 if result.status == "ok" else 6
        if args.three_d_command == "run":
            extra = None
            if args.extra_json:
                try:
                    parsed = json.loads(args.extra_json)
                except json.JSONDecodeError as exc:
                    print(f"--extra-json must be valid JSON: {exc}", file=sys.stderr)
                    return 2
                if not isinstance(parsed, dict):
                    print("--extra-json must decode to a JSON object", file=sys.stderr)
                    return 2
                extra = parsed
            request = ThreeDGenRequest(
                backend=args.backend,
                image_path=args.image_path,
                output_dir=args.output_dir,
                model_path=args.model_path,
                output_format=args.output_format,
                seed=args.seed,
                variant=args.variant,
                texture=args.texture,
                extra_args=extra,
            )
            try:
                env_overrides = {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices} if args.cuda_visible_devices else None
                result = generate_3d(
                    request,
                    dry_run=args.dry_run,
                    timeout_sec=args.timeout_sec,
                    env_overrides=env_overrides,
                )
            except (ThreeDGenBridgeError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(json.dumps(to_json(result), indent=2, sort_keys=True))
            return 0 if result.status in {"ok", "loaded", "dry_run"} else 5

    if args.command == "workers":
        registry = load_worker_registry()
        if args.workers_command == "list":
            for worker in registry.filter(lane=args.lane, env=args.env, status=args.status):
                models = ",".join(worker.models)
                expected_python = ",".join(worker.expected_python) or "unknown"
                print(
                    f"{worker.worker_id}\t{worker.lane}\tenv={worker.env}\t"
                    f"python={expected_python}\tstatus={worker.status}\t"
                    f"artifact_contract={worker.artifact_contract}\tmodels={models}"
                )
            return 0
        try:
            worker = registry.for_model(args.worker_or_model) if args.model else registry.get(args.worker_or_model)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.workers_command == "doctor":
            print(json.dumps(worker_to_json(worker), indent=2, sort_keys=True))
            return 0
        if args.workers_command == "preflight":
            result = preflight_worker(worker, timeout_sec=args.timeout_sec)
            print(json.dumps(worker_to_json(result), indent=2, sort_keys=True))
            return 0 if result.status == "ok" else 6

    if args.command == "cosmos25":
        if args.cosmos_command == "plan":
            extra = None
            if args.extra_json:
                try:
                    extra = json.loads(args.extra_json)
                except json.JSONDecodeError as exc:
                    print(f"--extra-json must be valid JSON: {exc}", file=sys.stderr)
                    return 2
                if not isinstance(extra, dict):
                    print("--extra-json must decode to a JSON object", file=sys.stderr)
                    return 2
            request = Cosmos25Request(
                model_id=args.model_id,
                input_files=tuple(args.input),
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint_path,
                repo_root=args.repo_root,
                model=args.model,
                inference_type=args.inference_type,
                nproc_per_node=args.nproc_per_node,
                master_port=args.master_port,
                context_parallel_size=args.context_parallel_size,
                cuda_visible_devices=args.cuda_visible_devices,
                offline_only=not args.allow_downloads,
                disable_guardrails=not args.enable_guardrails,
                offload_diffusion_model=args.offload_diffusion_model,
                offload_text_encoder=args.offload_text_encoder,
                offload_tokenizer=args.offload_tokenizer,
                student_only=args.student_only,
                inspect_only=True,
                extra_args=extra,
            )
            result = plan_or_run_cosmos25(request, dry_run=args.dry_run, timeout_sec=args.timeout_sec)
            print(json.dumps(cosmos25_to_json(result), indent=2, sort_keys=True))
            return 0 if result.status in {"ready", "dry_run"} else 8

    if args.command == "causal-lm":
        if args.causal_lm_command == "generate":
            trust_remote_code = None
            if args.trust_remote_code and args.no_trust_remote_code:
                print("choose only one of --trust-remote-code/--no-trust-remote-code", file=sys.stderr)
                return 2
            if args.trust_remote_code:
                trust_remote_code = True
            if args.no_trust_remote_code:
                trust_remote_code = False
            use_cache = None
            if args.use_cache and args.no_use_cache:
                print("choose only one of --use-cache/--no-use-cache", file=sys.stderr)
                return 2
            if args.use_cache:
                use_cache = True
            if args.no_use_cache:
                use_cache = False
            result = generate_causal_lm(
                CausalLMRequest(
                    model_id=args.model_id,
                    prompt=args.prompt,
                    model_root=args.model_root,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    dtype=args.dtype,
                    trust_remote_code=trust_remote_code,
                    use_cache=use_cache,
                    dry_run=args.dry_run,
                )
            )
            print(json.dumps(causal_lm_to_json(result), indent=2, sort_keys=True))
            return 0 if result.status in {"ok", "dry_run"} else 9

    if args.command == "classic-transformers":
        if args.classic_command == "inspect":
            result = inspect_classic_transformers(
                ClassicTransformersRequest(
                    model_id=args.model_id,
                    model_root=args.model_root,
                    task=args.task,
                    device=args.device,
                    dtype=args.dtype,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    run_synthetic=args.run_synthetic,
                    trust_remote_code=args.trust_remote_code,
                )
            )
            print(json.dumps(classic_transformers_to_json(result), indent=2, sort_keys=True))
            return 0 if result.status in {"ready", "ok"} else 10

    if args.command == "nemo-asr":
        if args.nemo_command == "run":
            request = NemoASRRequest(
                model_id=args.model_id,
                archive_path=args.archive_path,
                audio_paths=tuple(args.audio),
                output_dir=args.output_dir,
                map_location=args.map_location,
                device=args.device,
                restore=not args.inspect_only,
                load_only=args.load_only,
                force_restore=args.force_restore,
            )
            result = transcribe_nemo_asr(request, dry_run=args.dry_run, timeout_sec=args.timeout_sec)
            print(json.dumps(nemo_asr_to_json(result), indent=2, sort_keys=True))
            return 0 if result.status in {"ok", "loaded", "ready", "dry_run"} else 7

    if args.command == "generate-integration":
        try:
            plan = plan_integration_skeleton(
                args.model_id,
                catalog_path=args.catalog,
                out_dir=args.out_dir,
                name=args.name,
                status=args.status,
                include=tuple(args.include),
                allow_uncataloged=args.allow_uncataloged,
                lane=args.lane,
                preferred_env=args.preferred_env,
            )
            if args.dry_run:
                for path in plan.files:
                    print(path)
                return 0
            written = write_integration_skeleton(plan, force=args.force, test_style=args.test_style)
        except IntegrationSkeletonError as exc:
            print(str(exc), file=sys.stderr)
            return 4
        for path in written:
            print(path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
