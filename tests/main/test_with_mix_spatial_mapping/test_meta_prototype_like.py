import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when an incomplete spatial_mapping is provided and spatial_mapping_hint is also provided.

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "inputs/workload/alexnet.onnx": (5679695605, 8299150),
    "inputs/workload/mobilenetv2.onnx": (901092009, 2610609),
    "inputs/workload/resnet18.onnx": (1730672410, 3262009),
    "inputs/workload/resnet18.yaml": (2265438430, 4017227),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/meta_prototype_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/meta_prototype.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, _) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
