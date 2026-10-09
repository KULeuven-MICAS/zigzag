import yaml

from zigzag.api import get_hardware_performance_zigzag

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
    """The same 32 x 32 array, its operands broadcast or moving one unit per cycle along both dimensions, running a
    convolution with 64 output and 3 input channels over them: the systolic one drains across the 32 and the 3 units
    in use, 31 + 2 cycles after its last computation, and spends the same energy, the registers the operands pass
    through being the per-unit memories it already has."""
    workload = tmp_path / "conv.yaml"
    workload.write_text(yaml.safe_dump(CONV))
    data = yaml.safe_load(open("zigzag/inputs/hardware/tpu_like.yaml"))
    data["operational_array"]["systolic_dimensions"] = ["D1", "D2"]
    systolic = tmp_path / "tpu_like_systolic.yaml"
    systolic.write_text(yaml.safe_dump(data, sort_keys=False))

    results = {}
    for name, hardware in (("broadcast", "zigzag/inputs/hardware/tpu_like.yaml"), ("systolic", str(systolic))):
        _, _, cmes = get_hardware_performance_zigzag(
            str(workload), hardware, "zigzag/inputs/mapping/tpu_like.yaml", lpf_limit=4
        )
        results[name] = cmes[0][0]
    broadcast, systolic_cme = results["broadcast"], results["systolic"]
    assert broadcast.systolic_drain_cycle == 0
    assert systolic_cme.systolic_drain_cycle == 31 + 2
    assert systolic_cme.latency_total2 == broadcast.latency_total2 + 33
    assert systolic_cme.latency_total0 == broadcast.latency_total0
    assert systolic_cme.energy_total == broadcast.energy_total
