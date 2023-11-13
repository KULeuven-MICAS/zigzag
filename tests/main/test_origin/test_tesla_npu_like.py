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
    "zigzag/inputs/examples/workload/alexnet.onnx": (6131950030.816001, 8486444),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (1671933042.2130003, 2909436),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1863717063.505, 3395752),
    "zigzag.inputs.examples.workload.resnet18": (2375316568.8910007, 4082454),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.tesla_npu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Tesla_NPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
