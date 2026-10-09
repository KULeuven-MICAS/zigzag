import yaml

from zigzag.api import get_hardware_performance_zigzag

TPU_LIKE = "zigzag/inputs/hardware/tpu_like.yaml"
CONV = [
    {
        "id": 0,
        "name": "conv1",
        "operator_type": "Conv",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": ["ix=2*ox+1*fx", "iy=2*oy+1*fy"],
        "loop_dims": ["B", "K", "C", "OY", "OX", "FY", "FX"],
        "loop_sizes": [1, 64, 3, 112, 112, 7, 7],
        "operand_precision": {"W": 8, "I": 8, "O": 16, "O_final": 8},
        "operand_source": {"I": 0, "W": 0},
    }
]


def test_results_leave_a_systolic_array_a_cycle_per_unit_after_they_are_computed(tmp_path):
    """The tpu_like array, its operands moving one unit per cycle along both dimensions, and the same array with its
    operands broadcast, running a convolution with 64 output and 3 input channels over them: the systolic one drains
    across the 32 and the 3 units in use, 31 + 2 cycles after its last computation, and spends the same energy, the
    registers the operands pass through being the per-unit memories it already has."""
    workload = tmp_path / "conv.yaml"
    workload.write_text(yaml.safe_dump(CONV))
    with open(TPU_LIKE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["operational_array"].pop("systolic_dimensions") == ["D1", "D2"]
    broadcast = tmp_path / "tpu_like_broadcast.yaml"
    broadcast.write_text(yaml.safe_dump(data, sort_keys=False))

    results = {}
    for name, hardware in (("broadcast", str(broadcast)), ("systolic", TPU_LIKE)):
        _, _, cmes = get_hardware_performance_zigzag(
            str(workload), hardware, "zigzag/inputs/mapping/tpu_like.yaml", lpf_limit=4
        )
        results[name] = cmes[0][0]
    broadcast_cme, systolic_cme = results["broadcast"], results["systolic"]
    assert broadcast_cme.systolic_drain_cycle == 0
    assert systolic_cme.systolic_drain_cycle == 31 + 2
    assert systolic_cme.latency_total2 == broadcast_cme.latency_total2 + 33
    assert systolic_cme.latency_total0 == broadcast_cme.latency_total0
    assert systolic_cme.energy_total == broadcast_cme.energy_total
