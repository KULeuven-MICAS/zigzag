import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_without_unused_memory

workloads = (
    "zigzag/inputs/workload/alexnet.onnx",
    "zigzag/inputs/workload/mobilenetv2.onnx",
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (5568602396.684999, 8134431),
    "zigzag/inputs/workload/mobilenetv2.onnx": (751128562.4699999, 2427487),
    "zigzag/inputs/workload/resnet18.onnx": (1784539639.4349997, 3176546),
    "zigzag/inputs/workload/resnet18.yaml": (2115122870.395, 3884789),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.edge_tpu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Edge_TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
