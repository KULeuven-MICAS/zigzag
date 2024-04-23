import pytest
import sys

sys.path.append("../zigzag")
from zigzag.api import get_hardware_performance_zigzag_imc

workloads = (
    "zigzag/inputs/examples/workload/alexnet.onnx",
    "zigzag/inputs/examples/workload/mobilenetv2.onnx",
    "zigzag/inputs/examples/workload/resnet18.onnx",
    "zigzag.inputs.examples.workload.resnet18",
)

# Expected energy, latency (#cycles), clk time and area for each workload defined above
ens_lats_clks_areas = {
    "zigzag/inputs/examples/workload/alexnet.onnx": (2557076250.266322, 44012016.0, 6.61184, 0.7892517658006044),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (802185102.578702, 14939020.0, 6.61184, 0.7892517658006044),
    "zigzag/inputs/examples/workload/resnet18.onnx": (2252151728.145326, 62079022.0, 6.61184, 0.7892517658006044),
    "zigzag.inputs.examples.workload.resnet18": (2466090000.2577806, 67309272.0, 6.61184, 0.7892517658006044),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.default_imc"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Aimc"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, tclk, area, cmes) = get_hardware_performance_zigzag_imc(workload, accelerator, mapping)
    (expected_energy, expected_latency, expected_tclk, expected_area) = ens_lats_clks_areas[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
    assert tclk == pytest.approx(expected_tclk)
    assert area == pytest.approx(expected_area)
