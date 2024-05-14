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
    "zigzag/inputs/examples/workload/alexnet.onnx": (2340181787.2719307, 72692592.0, 3.2026, 0.785592664),
    "zigzag/inputs/examples/workload/mobilenetv2.onnx": (703506891.3687075, 28005964.0, 3.2026, 0.785592664),
    "zigzag/inputs/examples/workload/resnet18.onnx": (1828766840.9463186, 120700590.0, 3.2026, 0.785592664),
    "zigzag.inputs.examples.workload.resnet18": (2008581031.8287854, 130747736.0, 3.2026, 0.785592664),
}


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.default_imc"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.Dimc"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload, accelerator, mapping):
    (energy, latency, tclk, area, cmes) = get_hardware_performance_zigzag_imc(workload, accelerator, mapping)
    (expected_energy, expected_latency, expected_tclk, expected_area) = ens_lats_clks_areas[workload]
    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)
    assert tclk == pytest.approx(expected_tclk)
    assert area == pytest.approx(expected_area)
