import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_without_unused_memory

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/resnet18.onnx": (1659252422.016, 4000289),
    "zigzag/inputs/workload/resnet18.yaml": (1982830786.5119998, 4509235),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.tpu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
