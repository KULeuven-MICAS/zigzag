import pytest

from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when both spatial_mapping and spatial_mapping_hint are provided.

workloads = (
    "zigzag/inputs/examples/workload/alexnet.onnx",
    "zigzag/inputs/examples/workload/mobilenetv2.onnx",
    "zigzag/inputs/examples/workload/resnet18.onnx",
    "zigzag.inputs.examples.workload.resnet18",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/examples/workload/alexnet.onnx": (5582059481.445, 8343378),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (819971935.77, 2430583),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1763135800.67, 5001291),
    "zigzag.inputs.examples.workload.resnet18": (2090252961.0700002, 5858437),
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
            # spatial_mapping_hint will not work if the mapping on every dimension is provided in spatial_mapping
            "spatial_mapping_hint": {
                "D1": ["K"],
                "D2": ["C", "FX", "FY"],
                "D3": ["OX", "OY"],
                "D4": ["OX", "OY"],
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
    (energy, latency, cmes) = get_hardware_performance_zigzag_with_mix_spatial_mapping(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
