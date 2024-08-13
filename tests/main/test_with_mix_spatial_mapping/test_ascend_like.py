import pytest

from zigzag.api import get_hardware_performance_zigzag

# Test case for when spatial_mapping is provided, while spatial_mapping_hint is not provided.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (5664863934, 8479644),
    "zigzag/inputs/workload/mobilenetv2.onnx": (920740293, 3814440),
    "zigzag/inputs/workload/resnet18.onnx": (1680938520, 3697986),
    "zigzag/inputs/workload/resnet18.yaml": (2286320480, 4418088),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/ascend_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/ascend_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621
    energy, latency, _ = get_hardware_performance_zigzag(
        workload, accelerator, mapping, enable_mix_spatial_mapping=True
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
