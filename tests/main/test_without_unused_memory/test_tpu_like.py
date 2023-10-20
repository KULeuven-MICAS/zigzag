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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5475639384.492001, 8979956),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (952688145.0069999, 21873214),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1659252422.016, 4000289),
    "zigzag.inputs.examples.workload.resnet18": (1982830786.5119998, 4509235),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.tpu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
