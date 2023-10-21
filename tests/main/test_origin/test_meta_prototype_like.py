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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5771558839.89, 8400651),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (1731935837.864999, 3594631),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1869519792.3449998, 3408373),
    "zigzag.inputs.examples.workload.resnet18": (2419893343.4549994, 4176163),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.meta_prototype_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Meta_prototype"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
