import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when spatial_mapping is provided, while spatial_mapping_hint is not provided.

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "inputs/workload/alexnet.onnx": (5667407342.66, 8528846),
    "inputs/workload/mobilenetv2.onnx": (921552096.0700004, 3828967),
    "inputs/workload/resnet18.onnx": (1679218425.5100002, 3713386),
    "inputs/workload/resnet18.yaml": (2290766279.31, 4442443),
}


@pytest.fixture
def mapping():
    ascend_like_mapping = {
        "default": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("K", 16),
                "D2": (("C", 4), ("FX", 3)),
                "D3": ("OX", 2),
                "D4": ("OY", 2),
            },
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
        "Add": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("G", 16),
                "D2": ("C", 1),
                "D3": ("OX", 1),
                "D4": ("OY", 1),
            },
            "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        },
    }

    return ascend_like_mapping


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Ascend_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, cmes) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
