from scripts.derive_legacy_upgrade_candidates import infer_lane, infer_mismatches, score_model


def test_upgrade_candidate_infers_musicgen_audio_lane() -> None:
    model = {
        "id": "musicgen-small",
        "model_type": "musicgen",
        "library_name": "transformers",
        "example": {"snippet": "pip install --upgrade transformers scipy"},
    }

    candidate = score_model(model, set())

    assert candidate is not None
    assert candidate["lane"] == "audio_generation_bridge"
    assert "transformers" in candidate["dependency_hits"]


def test_upgrade_candidate_skips_cataloged_models() -> None:
    model = {"id": "musicgen-small", "model_type": "musicgen", "example": {"snippet": "transformers"}}

    assert score_model(model, {"musicgen-small"}) is None


def test_upgrade_candidate_mismatch_detection() -> None:
    model = {"id": "x", "example": {"snippet": "trust_remote_code=True flash_attn model.pt"}}

    mismatches = infer_mismatches(model, ["torch"])

    assert "remote_code_loader" in mismatches
    assert "attention_backend_selection" in mismatches
    assert "pytorch_checkpoint_contract" in mismatches


def test_upgrade_candidate_lane_defaults_to_transformers_auto() -> None:
    assert infer_lane({"id": "C-RADIOv4-SO400M", "model_type": None}, ["transformers"]) == "transformers_auto_bridge"
