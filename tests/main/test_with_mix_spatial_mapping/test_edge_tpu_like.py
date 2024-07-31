import pytest

from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when both spatial_mapping and spatial_mapping_hint are provided.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/resnet18.onnx": (1738892454.2700002, 4247189.0),
    "zigzag/inputs/workload/resnet18.yaml": (2029477205, 4738407),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/edge_tpu_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/edge_tpu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621
    (energy, latency, _) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
