import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when only spatial_mapping_hint is provided.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/resnet18.onnx": (1724869681, 3257898),
    "zigzag/inputs/workload/resnet18.yaml": (2220861655, 3934616),
}


@pytest.fixture
def mapping():
    tesla_npu_like_mapping = {
        "default": {
            "core_allocation": 1,
            # "spatial_mapping": {"D1": ("K", 32), "D2": ("OX", 8), "D3": ("OY", 4)},
            "spatial_mapping_hint": {"D1": ["K"], "D2": ["OX", "OY"]},
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
        "Add": {
            "core_allocation": 1,
            "spatial_mapping": {"D1": ("G", 32), "D2": ("OX", 1), "D3": ("OY", 1)},
            "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        },
        "Pooling": {
            "core_allocation": 1,
            "spatial_mapping": {"D1": ("G", 32), "D2": ("OX", 1), "D3": ("OY", 1)},
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
    }

    return tesla_npu_like_mapping


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Tesla_NPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, cmes) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
