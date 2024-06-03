import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when more non-existent dimensions are provided in spatial_mapping_hint.

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "inputs/workload/alexnet.onnx": (5468347515.336, 8210374),
    "inputs/workload/mobilenetv2.onnx": (730691567.0230001, 3358406),
    "inputs/workload/resnet18.onnx": (1648700710.272, 2946593),
    "inputs/workload/resnet18.yaml": (1972279074.768, 3455539),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/tpu_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/tpu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, _) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
