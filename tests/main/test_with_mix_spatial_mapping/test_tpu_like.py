import pytest

from zigzag.api import get_hardware_performance_zigzag_with_mix_spatial_mapping

# Test case for when more non-existent dimensions are provided in spatial_mapping_hint.

workloads = (
    "zigzag/inputs/examples/workload/alexnet.onnx",
    "zigzag/inputs/examples/workload/mobilenetv2.onnx",
    "zigzag/inputs/examples/workload/resnet18.onnx",
    "zigzag.inputs.examples.workload.resnet18",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/examples/workload/alexnet.onnx": (5468347515.336, 8210374),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (730691567.0230001, 3358406),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1648700710.272, 2946593),
    "zigzag.inputs.examples.workload.resnet18": (1972279074.768, 3455539),
}


@pytest.fixture
def mapping():
    tpu_like_mapping = {
        "default": {
            "core_allocation": 1,
            # "spatial_mapping": {
            #     "D1": ("K", 32),
            #     "D2": (("C", 2), ("FX", 3), ("FY", 3)),
            # },
            # D3 and D4 in spatial_mapping_hint will not work, since they do not exist in the hardware dimensions.
            "spatial_mapping_hint": {
                "D1": ["K"],
                "D2": ["C", "FX", "FY"],
                "D3": ["K", "OX"],
                "D4": ["OX", "OY"],
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
    (energy, latency, cmes) = get_hardware_performance_zigzag_with_mix_spatial_mapping(
        workload, accelerator, mapping
    )
    (expected_energy, expected_latency) = ens_lats[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
