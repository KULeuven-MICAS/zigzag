import pytest

from zigzag.api import get_hardware_performance_zigzag

workloads = ("zigzag/inputs/workload/gemm_layer.yaml",)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/gemm_layer.yaml": (1111828.48, 8194),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/gemm_l1_l3.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/gemm_l1_l3.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621
    energy, latency, _ = get_hardware_performance_zigzag(workload, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
