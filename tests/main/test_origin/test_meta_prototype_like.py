import pytest

from zigzag.api import get_hardware_performance_zigzag

workloads = (
    "zigzag/inputs/workload/alexnet.onnx",
    "zigzag/inputs/workload/mobilenetv2.onnx",
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (5771499135.4800005, 8338950.0),
    "zigzag/inputs/workload/mobilenetv2.onnx": (1728572789.1600003, 3429446.0),
    "zigzag/inputs/workload/resnet18.onnx": (1869036158.08, 3366695.0),
    "zigzag/inputs/workload/resnet18.yaml": (2418511845.2400002, 4130645.0),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/meta_prototype_like.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/meta_prototype.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, cmes) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
