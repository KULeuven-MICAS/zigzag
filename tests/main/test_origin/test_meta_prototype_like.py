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
    "inputs/workload/alexnet.onnx": (5766869514.52, 8338950),
    "inputs/workload/mobilenetv2.onnx": (1728388906.7599993, 3429446),
    "inputs/workload/resnet18.onnx": (1868963025.12, 3366695),
    "inputs/workload/resnet18.yaml": (2352271282.04, 4129027),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/meta_prototype_like.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/meta_prototype.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
