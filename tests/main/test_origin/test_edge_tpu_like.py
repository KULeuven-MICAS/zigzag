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
    "inputs/workload/alexnet.onnx": (5657178248.6, 8221391.0),
    "inputs/workload/mobilenetv2.onnx": (1680773377.4499998, 3562331.0),
    "inputs/workload/resnet18.onnx": (1902637139.1499994, 3333310.0),
    "inputs/workload/resnet18.yaml": (2413348670.67, 4268451.0),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/edge_tpu_like.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/edge_tpu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
