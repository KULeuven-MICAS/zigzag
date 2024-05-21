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
    "inputs/workload/alexnet.onnx": (6812722722.515776, 9460061.0, 3.75708, 0.8566212024),
    "inputs/workload/mobilenetv2.onnx": (2857471628.5898623, 20501552.0, 3.75708, 0.8566212024),
    "inputs/workload/resnet18.onnx": (1828766840.9463186, 120700590.0, 3.2026, 0.785592664),
    "inputs/workload/resnet18.yaml": (4268432908.954752, 5789353.0, 3.75708, 0.8566212024),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/default_imc.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/dimc.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):
    (energy, latency, tclk, area, cmes) = get_hardware_performance_zigzag_imc(workload, accelerator, mapping)
    (expected_energy, expected_latency, expected_tclk, expected_area) = ens_lats_clks_areas[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
    assert tclk == pytest.approx(expected_tclk)
    assert area == pytest.approx(expected_area)
