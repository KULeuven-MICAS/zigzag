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
    "inputs/workload/alexnet.onnx": (6811036226.435776, 9458913.0, 3.75708, 0.8566212024),
    "inputs/workload/mobilenetv2.onnx": (2884847085.7898636, 20453908.0, 3.75708, 0.8566212024),
    "inputs/workload/resnet18.onnx": (4726170712.825855, 6337728.0, 3.75708, 0.8566212024),
    "inputs/workload/resnet18.yaml": (4268285089.3547516, 5789229.0, 3.75708, 0.8566212024),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/default_imc.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/dimc.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, tclk, area, _) = get_hardware_performance_zigzag_imc(workload, accelerator, mapping)
    (expected_energy, expected_latency, expected_tclk, expected_area) = ens_lats_clks_areas[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
    assert tclk == pytest.approx(expected_tclk)
    assert area == pytest.approx(expected_area)
