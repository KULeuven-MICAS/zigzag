import pytest

from zigzag.api import get_hardware_performance_zigzag

workloads = (
    "zigzag/inputs/examples/workload/alexnet.onnx",
    "zigzag/inputs/examples/workload/mobilenetv2.onnx",
    "zigzag/inputs/examples/workload/resnet18.onnx",
    "zigzag.inputs.examples.workload.resnet18",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/examples/workload/alexnet.onnx": (5562971337.551999, 9061821),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (1904302988.3070006, 23101112),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1795832911.4720004, 4158539),
    "zigzag.inputs.examples.workload.resnet18": (2230898567.856, 4816575),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.tpu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, _) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
