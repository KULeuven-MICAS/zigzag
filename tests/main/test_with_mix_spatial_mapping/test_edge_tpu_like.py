import pytest

from zigzag.api import get_hardware_performance_zigzag_without_unused_memory

workloads = (
    "zigzag/inputs/examples/workload/alexnet.onnx",
    "zigzag/inputs/examples/workload/mobilenetv2.onnx",
    "zigzag/inputs/examples/workload/resnet18.onnx",
    "zigzag.inputs.examples.workload.resnet18",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/examples/workload/alexnet.onnx": (5568602396.684999, 8134431),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (751128562.4699999, 2427487),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1784539639.4349997, 3176546),
    "zigzag.inputs.examples.workload.resnet18": (2115122870.395, 3884789),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.edge_tpu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Edge_TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
