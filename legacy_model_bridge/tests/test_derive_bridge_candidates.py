import json
import subprocess
import sys
from pathlib import Path


def test_derive_bridge_candidates_from_small_index(tmp_path: Path) -> None:
    model_index = tmp_path / "model_index.json"
    model_index.write_text(
        json.dumps(
            {
                "generated_utc": "2026-07-15T17:30:42Z",
                "model_count": 2,
                "models": [
                    {
                        "id": "example-diffusers",
                        "library_name": "diffusers",
                        "pipeline_tag": "text-to-video",
                        "model_type": None,
                        "tree_weight_size_bytes": 10,
                        "tree_file_extensions": {".safetensors": 1},
                    },
                    {
                        "id": "example-nemo",
                        "library_name": "nemo",
                        "pipeline_tag": "automatic-speech-recognition",
                        "model_type": None,
                        "tree_weight_size_bytes": 5,
                        "tree_file_extensions": {".nemo": 1},
                    },
                ],
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/derive_bridge_candidates.py",
            "--model-index",
            str(model_index),
            "--limit",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)

    assert payload["metadata"]["source_model_count"] == 2
    assert payload["metadata"]["lane_counts"]["diffusers_cuda_bridge"] == 1
    assert payload["metadata"]["lane_counts"]["nemo_asr_bridge"] == 1
