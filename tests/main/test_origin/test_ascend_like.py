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
    "inputs/workload/alexnet.onnx": (5738188827.200001, 8696068.0),
    "inputs/workload/mobilenetv2.onnx": (1913154775.600001, 7359658.0),
    "inputs/workload/resnet18.onnx": (1860963861.6800003, 3698589.0),
    "inputs/workload/resnet18.yaml": (2411783423.28, 4779709.0),
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
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
