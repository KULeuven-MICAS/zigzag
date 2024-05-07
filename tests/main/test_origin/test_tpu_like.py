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
    "inputs/workload/alexnet.onnx": (5562971337.551999, 9061821),
    "inputs/workload/mobilenetv2.onnx": (1904302988.3070006, 23101112),
    "inputs/workload/resnet18.onnx": (1795832911.4720004, 4158539),
    "inputs/workload/resnet18.yaml": (2230898567.856, 4816575),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/tpu_like.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/tpu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, _) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
