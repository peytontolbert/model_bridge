import pytest

from legacy_model_bridge.runtime.encoder_classifier import (
    EncoderClassifierBridgeError,
    classifier_head_shape,
    generic_label_maps,
    repair_config_from_classifier_head,
)


class ShapeOnly:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class Config:
    num_labels = 2
    id2label = {0: "NEG", 1: "POS"}
    label2id = {"NEG": 0, "POS": 1}


def test_classifier_head_shape_reads_weight_and_bias() -> None:
    head = classifier_head_shape(
        {
            "classifier.weight": ShapeOnly((4, 768)),
            "classifier.bias": ShapeOnly((4,)),
        }
    )

    assert head.num_labels == 4
    assert head.hidden_size == 768
    assert head.has_bias is True


def test_classifier_head_shape_requires_weight() -> None:
    with pytest.raises(EncoderClassifierBridgeError, match="missing required tensor"):
        classifier_head_shape({})


def test_classifier_head_shape_rejects_rank_mismatch() -> None:
    with pytest.raises(EncoderClassifierBridgeError, match="rank 2"):
        classifier_head_shape({"classifier.weight": ShapeOnly((4, 768, 1))})


def test_classifier_head_shape_rejects_bias_mismatch() -> None:
    with pytest.raises(EncoderClassifierBridgeError, match="does not match"):
        classifier_head_shape(
            {
                "classifier.weight": ShapeOnly((4, 768)),
                "classifier.bias": ShapeOnly((3,)),
            }
        )


def test_generic_label_maps_use_repaired_cardinality() -> None:
    id2label, label2id = generic_label_maps(3)

    assert id2label == {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    assert label2id == {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}


def test_repair_config_from_classifier_head_overrides_stale_labels() -> None:
    config = Config()

    repaired = repair_config_from_classifier_head(config, {"classifier.weight": ShapeOnly((5, 384))})

    assert repaired is config
    assert repaired.num_labels == 5
    assert repaired.id2label[4] == "LABEL_4"
    assert repaired.label2id["LABEL_4"] == 4
