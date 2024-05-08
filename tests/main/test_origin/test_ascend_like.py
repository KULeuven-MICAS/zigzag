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
    "inputs/workload/alexnet.onnx": (5737868753.12, 8696023),
    "inputs/workload/mobilenetv2.onnx": (1913561726.0000005, 7359650),
    "inputs/workload/resnet18.onnx": (1860918012.2400002, 3698589),
    "inputs/workload/resnet18.yaml": (2345967030.96, 4779555),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/ascend_like.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/ascend_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
