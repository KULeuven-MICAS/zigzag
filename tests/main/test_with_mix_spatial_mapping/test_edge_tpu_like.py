import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when both spatial_mapping and spatial_mapping_hint are provided.

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "inputs/workload/alexnet.onnx": (5582059481.445, 8343378),
    "inputs/workload/mobilenetv2.onnx": (819971935.77, 2430583),
    "inputs/workload/resnet18.onnx": (1763135800.67, 5001291),
    "inputs/workload/resnet18.yaml": (2090252961.0700002, 5858437),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/edge_tpu_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/edge_tpu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, _) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
