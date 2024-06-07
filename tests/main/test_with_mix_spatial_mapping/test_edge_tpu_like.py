import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when both spatial_mapping and spatial_mapping_hint are provided.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (6159980160, 8337351),
    "zigzag/inputs/workload/mobilenetv2.onnx": (742114179, 2421959),
    "zigzag/inputs/workload/resnet18.onnx": (1735517944, 4055269),
    "zigzag/inputs/workload/resnet18.yaml": (2029477205, 4738407),
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
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
