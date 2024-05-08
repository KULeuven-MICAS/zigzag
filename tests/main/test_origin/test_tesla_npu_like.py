import pytest

from zigzag.api import get_hardware_performance_zigzag

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "inputs/workload/alexnet.onnx": (6129156613.576, 8442657),
    "inputs/workload/mobilenetv2.onnx": (1682873656.7980008, 2824122),
    "inputs/workload/resnet18.onnx": (1863651442.3999999, 3380242),
    "inputs/workload/resnet18.yaml": (2308838375.536, 4066942),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/tesla_npu_like.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/tesla_npu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
