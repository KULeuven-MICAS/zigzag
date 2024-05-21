import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_without_unused_memory

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "inputs/workload/alexnet.onnx": (6040086796.366001, 8389669),
    "inputs/workload/mobilenetv2.onnx": (930702060.6110002, 1965457),
    "inputs/workload/resnet18.onnx": (1724869681.4799998, 3257898),
    "inputs/workload/resnet18.yaml": (2220861655.6660004, 3934616),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.tesla_npu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Tesla_NPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
