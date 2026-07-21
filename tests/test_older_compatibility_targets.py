import json
from pathlib import Path


def test_older_compatibility_targets_are_ranked_and_actionable() -> None:
    payload = json.loads(Path("data/older_compatibility_targets.json").read_text())

    ranks = [target["rank"] for target in payload["targets"]]
    assert ranks == sorted(ranks)
    assert payload["source_index"].endswith("model_index.json")
    assert payload["targets"][0]["lane"] == "nemo_asr_bridge"
    assert "parakeet-tdt_ctc-110m" in payload["targets"][0]["models"]
    assert all(target["first_smoke"] for target in payload["targets"])
    assert all(target["mismatch_classes"] for target in payload["targets"])
