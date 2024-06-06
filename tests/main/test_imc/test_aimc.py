import pytest

from zigzag.api import get_hardware_performance_zigzag_imc

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy, latency (#cycles), clk time and area for each workload defined above
ens_lats_clks_areas = {
    "inputs/workload/alexnet.onnx": (6780119394.667442, 9261169.0, 14.07244, 2.3426222152749223),
    "inputs/workload/mobilenetv2.onnx": (2318251910.682223, 10901796.0, 14.07244, 2.3426222152749223),
    "inputs/workload/resnet18.onnx": (4695729577.211851, 5540648.0, 14.07244, 2.3426222152749223),
    "inputs/workload/resnet18.yaml": (4264976904.6396008, 4907495.0, 14.07244, 2.3426222152749223),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/default_imc.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/aimc.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, tclk, area, cmes) = get_hardware_performance_zigzag_imc(workload, accelerator, mapping)
    (expected_energy, expected_latency, expected_tclk, expected_area) = ens_lats_clks_areas[workload]
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
    assert tclk == pytest.approx(expected_tclk)
    assert area == pytest.approx(expected_area)
