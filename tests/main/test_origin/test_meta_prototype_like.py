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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5766869514.52, 8338950),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (1728388906.7599993, 3429446),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1868963025.12, 3366695),
    "zigzag.inputs.examples.workload.resnet18": (2352271282.04, 4129027),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.meta_prototype_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Meta_prototype"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
