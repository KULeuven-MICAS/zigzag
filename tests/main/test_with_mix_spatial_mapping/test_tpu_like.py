import pytest

from zigzag.api import get_hardware_performance_zigzag

# Test case for when more non-existent dimensions are provided in spatial_mapping_hint.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (5589583403, 8671208),
    "zigzag/inputs/workload/mobilenetv2.onnx": (932571650, 7304307),
    "zigzag/inputs/workload/resnet18.onnx": (1759424218.3160002, 4495469.0),
    "zigzag/inputs/workload/resnet18.yaml": (2191903732.78, 4845353.0),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/tpu_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/tpu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621  # pylint: disable=W0621
    energy, latency, _ = get_hardware_performance_zigzag(
        workload, accelerator, mapping, enable_mix_spatial_mapping=True
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
