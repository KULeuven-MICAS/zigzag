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
    "inputs/workload/alexnet.onnx": (6781805890.747441, 9262317.0, 14.07244, 2.3426222152749223),
    "inputs/workload/mobilenetv2.onnx": (2290876453.4822226, 10949440.0, 14.07244, 2.3426222152749223),
    "inputs/workload/resnet18.onnx": (4726270705.225856, 6337852.0, 3.75708, 0.8566212024),
    "inputs/workload/resnet18.yaml": (4265124724.2396007, 4907619.0, 14.07244, 2.3426222152749223),
}


@pytest.fixture
def mapping():
    return "inputs/mapping/default_imc.yaml"


@pytest.fixture
def accelerator():
    return "inputs/hardware/aimc.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, tclk, area, cmes) = get_hardware_performance_zigzag_imc(workload, accelerator, mapping)
    (expected_energy, expected_latency, expected_tclk, expected_area) = ens_lats_clks_areas[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
    assert tclk == pytest.approx(expected_tclk)
    assert area == pytest.approx(expected_area)
