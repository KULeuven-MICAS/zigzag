# Test the zigzag api call
import pytest

from zigzag.api import get_hardware_performance_zigzag


@pytest.fixture
def workload():
    return "zigzag/inputs/examples/workload/mobilenetv2.onnx"


@pytest.fixture
def mapping():
    return "zigzag.inputs.examples.mapping.tpu_like"


@pytest.fixture
def accelerator():
    return "zigzag.inputs.examples.hardware.TPU_like"


def test_api(workload, mapping, accelerator):
    answer = get_hardware_performance_zigzag(workload, accelerator, mapping)
    assert answer
