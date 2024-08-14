import pytest

from zigzag.api import get_hardware_performance_zigzag_imc

workloads = (
    "zigzag/inputs/workload/resnet18.onnx",
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy, latency (#cycles), clk time and area for each workload defined above
ens_lats_clks_areas = {
    "zigzag/inputs/workload/resnet18.onnx": (
        4695729577.211851,
        5540648.0,
        14.07244,
        2.2878123152749223,
    ),
    "zigzag/inputs/workload/resnet18.yaml": (
        4264976904.6396008,
        4907495.0,
        14.07244,
        2.2878123152749223,
    ),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/default_imc.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/aimc.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621
    energy, latency, tclk, area, _ = get_hardware_performance_zigzag_imc(workload, accelerator, mapping)
    expected_energy, expected_latency, expected_tclk, expected_area = ens_lats_clks_areas[workload]
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
    assert tclk == pytest.approx(expected_tclk)  # type: ignore
    assert area == pytest.approx(expected_area)  # type: ignore
