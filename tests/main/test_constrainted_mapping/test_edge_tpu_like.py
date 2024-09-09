from zigzag.api import get_hardware_performance_zigzag
from zigzag.visualization.results.print_mapping import get_temporal_spatial_loops

# Test case for when both spatial_mapping and spatial_mapping_hint are provided.

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/alexnet.onnx": (6159980160, 8337351),
    "zigzag/inputs/workload/mobilenetv2.onnx": (742114179, 2421959),
    "zigzag/inputs/workload/resnet18.onnx": (1735517944, 4055269),
    "zigzag/inputs/workload/resnet18.yaml": (2029477205, 4738407),
}


# @pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/edge_tpu_like_constrainted.yaml"


# @pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/edge_tpu_like.yaml"


# @pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621
    energy, latency, cmes = get_hardware_performance_zigzag(workload, accelerator, mapping)
    temp_map_default = [get_temporal_spatial_loops(cme[0])[0] for cme in cmes[0][1]]

    for temp_mapping in temp_map_default:
        if ("OX" in [dim[0].name for dim in temp_mapping]) and ("OY" in [dim[0].name for dim in temp_mapping]):
            assert temp_mapping[-2][0].name == "OX"
            assert temp_mapping[-1][0].name == "OY"


test_api(workload=workloads[0], accelerator=accelerator(), mapping=mapping())
