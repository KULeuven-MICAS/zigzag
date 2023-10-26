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
    "zigzag/inputs/examples/workload/alexnet.onnx": (5589583592.816001, 8694888),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (933599264.2320005, 7309493),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1685730150.7640002, 4510061),
    "zigzag.inputs.examples.workload.resnet18": (2098653635.0679998, 4849417),
}


@pytest.fixture
def mapping():
    tpu_like_mapping = {
        "default": {
            "core_allocation": 1,
            "spatial_mapping": {
                "D1": ("K", 32),
                "D2": (("C", 2), ("FX", 3), ("FY", 3)),
            },
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
        "Add": {
            "core_allocation": 1,
            "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
            "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        },
        "Pooling": {
            "core_allocation": 1,
            "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        },
    }

    return tpu_like_mapping


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.TPU_like"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, cmes) = get_hardware_performance_zigzag_without_unused_memory(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
