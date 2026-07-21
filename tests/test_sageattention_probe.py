import sys
import types

from legacy_model_bridge.runtime.sageattention import sageattention_status


def test_sageattention_status_recommends_rebuild_when_symbols_missing(monkeypatch) -> None:
    fake_module = types.SimpleNamespace()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "sageattention" else None)
    monkeypatch.setitem(sys.modules, "sageattention", fake_module)

    status = sageattention_status()

    assert status.available is True
    assert status.upgrade_recommendation == "rebuild_sageattention_from_source"
    assert status.recommended_abot_backend == "sdpa"


def test_sageattention_status_recommends_keep_for_rtx3090(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(
        sageattn_qk_int8_pv_fp16_triton=object(),
        sageattn_qk_int8_pv_fp16_cuda=object(),
        sageattn_qk_int8_pv_fp8_cuda=object(),
        sageattn_varlen=object(),
    )

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(device):
            return "NVIDIA GeForce RTX 3090"

        @staticmethod
        def get_device_capability(device):
            return (8, 6)

    fake_torch = types.SimpleNamespace(__version__="2.10.0+cu128", version=types.SimpleNamespace(cuda="12.8"), cuda=FakeCuda)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "sageattention" else None)
    monkeypatch.setitem(sys.modules, "sageattention", fake_module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = sageattention_status()

    assert status.upgrade_recommendation == "do_not_upgrade_for_rtx3090"
    assert status.recommended_abot_backend == "sageattn"
