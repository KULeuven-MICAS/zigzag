import pytest

from zigzag.api import get_hardware_performance_zigzag

# Test case for when an incomplete spatial_mapping is provided and spatial_mapping_hint is also provided.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (5681909132.480001, 8279495.0),
    "zigzag/inputs/workload/mobilenetv2.onnx": (909647916.68, 2602479.0),
    "zigzag/inputs/workload/resnet18.onnx": (1751779924.0000002, 3234867.0),
    "zigzag/inputs/workload/resnet18.yaml": (2259198622.68, 3884859.0),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/meta_prototype_like_mixed.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/meta_prototype.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621
    energy, latency, _ = get_hardware_performance_zigzag(
        workload, accelerator, mapping, enable_mix_spatial_mapping=True
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
