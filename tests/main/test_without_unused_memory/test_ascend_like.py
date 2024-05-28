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
    "inputs/workload/alexnet.onnx": (5649555894.9, 8637780),
    "inputs/workload/mobilenetv2.onnx": (1881386179.71, 6486685),
    "inputs/workload/resnet18.onnx": (1709089377.83, 3583047),
    "inputs/workload/resnet18.yaml": (2243493483.15, 4657130),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.ascend_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Ascend_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
