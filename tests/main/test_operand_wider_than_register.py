import pytest

from zigzag.api import get_hardware_performance_zigzag
from zigzag.opt.loma.engine import NoValidLoopOrderingFoundException


def test_an_operand_wider_than_its_register_is_reported_as_not_fitting():
    """The Gemm's 32-bit partial sums do not fit the TPU-like 16-bit output register: the spatial mapping keeps its
    unrolling of one, and the search reports that the operand does not fit the lowest memory level."""
    with pytest.raises(NoValidLoopOrderingFoundException):
        get_hardware_performance_zigzag(
            "zigzag/inputs/workload/gemm_layer.yaml",
            "zigzag/inputs/hardware/tpu_like.yaml",
            "zigzag/inputs/mapping/tpu_like.yaml",
        )
