from legacy_model_bridge.consolidation import load_consolidation_plan


def test_consolidation_plan_loads_non_default_env_entries() -> None:
    plan = load_consolidation_plan()

    trellis = plan.get("microsoft/TRELLIS.2-4B")

    assert trellis.current_env == "trellis_deleted"
    assert trellis.requires_worker_boundary is False


def test_consolidation_counts_decisions() -> None:
    plan = load_consolidation_plan()

    counts = plan.decision_counts()

    assert counts["worker_boundary_required"] >= 2
    assert counts["latest_env_ready"] >= 4


def test_consolidation_filters_nemo_asr_into_ai() -> None:
    plan = load_consolidation_plan()

    entries = [entry for entry in plan.filter(current_env="ai") if entry.lane == "nemo_asr_bridge"]

    assert {entry.model_id for entry in entries} >= {"parakeet-rnnt-0.6b", "parakeet-tdt_ctc-110m"}


def test_consolidation_identifies_latest_env_candidates() -> None:
    plan = load_consolidation_plan()

    ids = {entry.model_id for entry in plan.removable_env_candidates()}

    assert "acvlab--ABot-World-0-5B-LF" in ids
    assert "Cosmos-Embed1-448p-anomaly-detection" in ids


def test_consolidation_records_caller_and_backend_python_separately() -> None:
    plan = load_consolidation_plan()

    trellis = plan.get("microsoft/TRELLIS.2-4B")
    nemo = plan.get("parakeet-rnnt-0.6b")

    assert trellis.supports_caller_python("3.12.1") is True
    assert trellis.backend_python == ("3.11", "3.12")
    assert "python_abi_or_native_extension" in trellis.mismatch_classes
    assert nemo.supports_caller_python("3.11") is True
    assert nemo.backend_python == ("3.11", "3.12")


def test_consolidation_filters_by_caller_python() -> None:
    plan = load_consolidation_plan()

    py312_entries = plan.filter(caller_python="3.12")

    assert {entry.model_id for entry in py312_entries} == {entry.model_id for entry in plan.entries}
    assert plan.filter(caller_python="3.13") == []


def test_caller_python_matrix_counts_supported_versions() -> None:
    plan = load_consolidation_plan()

    matrix = plan.caller_python_matrix()

    assert len(matrix["3.11"]) == len(plan.entries)
    assert len(matrix["3.12"]) == len(plan.entries)
