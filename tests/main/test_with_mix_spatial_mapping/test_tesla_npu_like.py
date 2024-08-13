import pytest

from zigzag.api import get_hardware_performance_zigzag

# Test case for when only spatial_mapping_hint is provided.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (6006136982.778, 8290892.0),
    "zigzag/inputs/workload/mobilenetv2.onnx": (947736166.5380002, 1857838.0),
    "zigzag/inputs/workload/resnet18.onnx": (1604556365.552, 2828301.0),
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
    energy, latency, _ = get_hardware_performance_zigzag(
        workload, accelerator, mapping, enable_mix_spatial_mapping=True
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
