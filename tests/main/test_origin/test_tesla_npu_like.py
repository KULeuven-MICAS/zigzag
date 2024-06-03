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
    "zigzag/inputs/workload/alexnet.onnx": (6131856327.976001, 8442657.0),
    "zigzag/inputs/workload/mobilenetv2.onnx": (1683001389.998, 2824122.0),
    "zigzag/inputs/workload/resnet18.onnx": (1863716799.84, 3380242.0),
    "zigzag/inputs/workload/resnet18.yaml": (2374655424.176, 4066942.0),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/tesla_npu_like.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/tesla_npu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, cmes) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
