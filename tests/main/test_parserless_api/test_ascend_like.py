import pytest
from zigzag.utils import open_yaml

from zigzag.api import get_hardware_performance_zigzag

workloads = (
    "zigzag/inputs/workload/resnet18.yaml",
)

# Expected energy and latency for each workload defined above
ens_lats = {
    "zigzag/inputs/workload/resnet18.yaml": (2449578483.76, 3980000.0),
}


@pytest.fixture
def mapping():
    return "zigzag/inputs/mapping/ascend_like.yaml"


@pytest.fixture
def accelerator():
    return "zigzag/inputs/hardware/ascend_like.yaml"


@pytest.mark.parametrize("workload", workloads)
def test_api(workload: str, accelerator: str, mapping: str):  # pylint: disable=W0621
    # This test uses the yaml data to define the workload directly,
    # so it can be defined by external tools without writing to a file
    data = open_yaml(workload)
    energy, latency, _ = get_hardware_performance_zigzag(data, accelerator, mapping)
    (expected_energy, expected_latency) = ens_lats[workload]
    print(f"'{workload}': ({energy}, {latency}),")
    assert energy == pytest.approx(expected_energy)  # type: ignore
    assert latency == pytest.approx(expected_latency)  # type: ignore
