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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5681909351.240001, 8299150),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (919452681.2249999, 2894129),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1789888904.4450002, 3472280),
    "zigzag.inputs.examples.workload.resnet18": (2348207081.7949996, 4238517),
}


@pytest.fixture
def mapping():
    meta_prototype_like_mapping = {
        "default": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("K", 32),
                "D2": ("C", 2),
                "D3": (("OX", 2), ("OY", 2)),
                "D4": (("OX", 2), ("OY", 2)),
            },
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
        "Add": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("G", 32),
                "D2": ("C", 1),
                "D3": ("OX", 1),
                "D4": ("OY", 1),
            },
            "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        },
    }
    return meta_prototype_like_mapping


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
