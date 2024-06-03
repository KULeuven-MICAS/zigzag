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
    return "inputs/mapping/ascend_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/ascend_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, _) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
