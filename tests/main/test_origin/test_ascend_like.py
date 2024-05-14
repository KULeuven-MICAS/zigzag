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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5737868753.12, 8696023),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (1913561726.0000005, 7359650),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1860918012.2400002, 3698589),
    "zigzag.inputs.examples.workload.resnet18": (2345967030.96, 4779555),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.ascend_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Ascend_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
