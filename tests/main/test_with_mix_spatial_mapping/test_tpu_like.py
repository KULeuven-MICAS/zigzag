import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when more non-existent dimensions are provided in spatial_mapping_hint.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/resnet18.onnx": (1648700710.272, 2946593),
    "zigzag/inputs/workload/resnet18.yaml": (1972279074.768, 3455539),
}


@pytest.fixture
def mapping():
    tpu_like_mapping = {
        "default": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("K", 32),
                "D2": (("C", 2), ("FX", 3), ("FY", 3)),
            },
            "spatial_mapping_hint": {
                "D1": ["K"],
                "D2": ["C", "FX", "FY"],
                "D3": ["K", "OX"],
                "D4": ["OX", "OY"],
            },
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
        "Add": {
            "core_allocation": 1,
            "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
            "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        },
        "Pooling": {
            "core_allocation": 1,
            "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
    }

    return tpu_like_mapping


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, cmes) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
