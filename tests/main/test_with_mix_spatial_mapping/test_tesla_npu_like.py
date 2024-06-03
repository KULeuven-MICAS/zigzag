import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when only spatial_mapping_hint is provided.

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "inputs/workload/alexnet.onnx": (6044768678, 8370470),
    "inputs/workload/mobilenetv2.onnx": (930702060, 1965457),
    "inputs/workload/resnet18.onnx": (1724869681, 3257898),
    "inputs/workload/resnet18.yaml": (2220861655, 3934616),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/tesla_npu_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/tesla_npu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, _) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
