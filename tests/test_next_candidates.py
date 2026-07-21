from legacy_model_bridge.next_candidates import load_next_integration_plan, to_json


def test_next_integration_plan_loads_ranked_candidates() -> None:
    plan = load_next_integration_plan()

    assert len(plan.candidates) >= 8
    assert plan.candidates[0].model_id == "nvidia/Cosmos-Transfer2.5-2B"
    assert plan.candidates[0].rank == 1
    assert plan.candidates[0].caller_python == ("3.11", "3.12")


def test_next_integration_plan_filters_by_lane_and_env() -> None:
    plan = load_next_integration_plan()

    cosmos = plan.filter(lane="cosmos25_official_bridge", env="cosmos25_py310")

    assert [candidate.model_id for candidate in cosmos] == [
        "nvidia/Cosmos-Transfer2.5-2B",
        "nvidia/Cosmos-Predict2.5-14B",
    ]


def test_next_integration_candidate_json_roundtrip_shape() -> None:
    candidate = load_next_integration_plan().get("MobileLLM family")
    payload = to_json(candidate)

    assert payload["lane"] == "transformers_causal_lm_bridge"
    assert "legacy_tokenizer_fallback" in payload["mismatch_classes"]
    assert payload["first_smoke"]
