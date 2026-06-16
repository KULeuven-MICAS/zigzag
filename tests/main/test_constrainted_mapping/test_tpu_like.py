import pytest

from zigzag.api import get_hardware_performance_zigzag
from zigzag.visualization.results.print_mapping import get_temporal_spatial_loops

# Test case for when more non-existent dimensions are provided in spatial_mapping_hint.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (5589583403, 8671208),
    "zigzag/inputs/workload/mobilenetv2.onnx": (932571650, 7304307),
    "zigzag/inputs/workload/resnet18.onnx": (1759424218.3160002, 4495469.0),
    "zigzag/inputs/workload/resnet18.yaml": (2191903732.78, 4845353.0),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/tpu_like_constrainted.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/tpu_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621  # pylint: disable=W0621
    energy, latency, cmes = get_hardware_performance_zigzag(workload, accelerator, mapping)
    temp_map_default = [get_temporal_spatial_loops(cme[0])[0] for cme in cmes[0][1]]

    for temp_mapping in temp_map_default:
        if ("OX" in [dim[0].name for dim in temp_mapping]) and ("OY" in [dim[0].name for dim in temp_mapping]):
            assert temp_mapping[-2][0].name == "OX"
            assert temp_mapping[-1][0].name == "OY"
