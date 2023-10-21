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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5657179533.855, 8256774),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (1694757317.6950002, 3722532),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1903877328.0199995, 3363983),
    "zigzag.inputs.examples.workload.resnet18": (2413350265.7900004, 4314851),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.edge_tpu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Edge_TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
