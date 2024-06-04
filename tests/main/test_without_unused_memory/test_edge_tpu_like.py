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
    "inputs/workload/alexnet.onnx": (5568602396.684999, 8134431),
    "inputs/workload/mobilenetv2.onnx": (735250234.1699998, 2417321.0),
    "inputs/workload/resnet18.onnx": (1783299827.71, 3156625.0),
    "inputs/workload/resnet18.yaml": (2115121959.8699996, 3855157.0),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/edge_tpu_like.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/edge_tpu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, _) = get_hardware_performance_zigzag_without_unused_memory(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
