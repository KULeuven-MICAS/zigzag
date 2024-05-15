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
    "inputs/workload/alexnet.onnx": (5567501203.632, 9061821.0),
    "inputs/workload/mobilenetv2.onnx": (1904482765.907, 23101112.0),
    "inputs/workload/resnet18.onnx": (1795904402.5120003, 4158539.0),
    "inputs/workload/resnet18.yaml": (2296490149.296, 4906975.0),
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
    print(f"{workload}: ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
