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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5679695605.4400015, 8299150),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (901092009.6000001, 2610609),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1730672410.3200004, 3262009),
    "zigzag.inputs.examples.workload.resnet18": (2265438430.2299995, 4017227),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.meta_prototype_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Meta_prototype"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
