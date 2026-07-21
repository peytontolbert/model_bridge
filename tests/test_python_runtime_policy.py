import json
import tomllib
from pathlib import Path

from legacy_model_bridge.consolidation import load_consolidation_plan
from legacy_model_bridge.registry import load_catalog


def test_project_metadata_matches_caller_python_policy() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    project = pyproject["project"]

    assert project["requires-python"] == ">=3.11,<3.13"
    assert "Programming Language :: Python :: 3.11" in project["classifiers"]
    assert "Programming Language :: Python :: 3.12" in project["classifiers"]


def test_catalog_and_consolidation_share_caller_python_policy() -> None:
    catalog_raw = json.loads(Path("data/bridge_catalog.json").read_text())
    consolidation_raw = json.loads(Path("data/environment_consolidation.json").read_text())

    assert catalog_raw["metadata"]["caller_python_supported"] == ["3.11", "3.12"]
    assert consolidation_raw["metadata"]["caller_python_supported"] == ["3.11", "3.12"]

    catalog = load_catalog()
    plan = load_consolidation_plan()

    assert all(entry.caller_python == ("3.11", "3.12") for entry in catalog.entries)
    assert all(entry.caller_python == ("3.11", "3.12") for entry in plan.entries)
