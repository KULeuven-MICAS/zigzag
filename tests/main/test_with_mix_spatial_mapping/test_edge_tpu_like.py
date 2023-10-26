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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5582430184.085, 8343378),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (762066732.5049998, 3003074),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1743190534.155, 5305825),
    "zigzag.inputs.examples.workload.resnet18": (2087322696.315, 6155355),
}


@pytest.fixture
def mapping():
    edge_tpu_like_mapping = {
        "default": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("K", 8),
                "D2": (("C", 2), ("FX", 2), ("FY", 2)),
                "D3": (("OX", 2), ("OY", 2)),
                "D4": (("OX", 2), ("OY", 2)),
            },
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
        "Add": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("G", 8),
                "D2": ("C", 1),
                "D3": ("OX", 1),
                "D4": ("OY", 1),
            },
            "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        },
        "Pooling": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("G", 8),
                "D2": ("C", 1),
                "D3": ("OX", 1),
                "D4": ("OY", 1),
            },
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
    }
    return edge_tpu_like_mapping


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Edge_TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
