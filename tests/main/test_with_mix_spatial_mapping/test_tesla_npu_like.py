import pytest

from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when only spatial_mapping_hint is provided.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/resnet18.onnx": (1605857758.544, 3324427.0),
    "zigzag/inputs/workload/resnet18.yaml": (2094141825.5040002, 3480232.0),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/tesla_npu_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/tesla_npu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621
    (energy, latency, _) = get_hardware_performance_zigzag_with_mix_spatial_mapping(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
