from typing import Any

import pytest

from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.opt.loma.engine import NoValidLoopOrderingFoundException
from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.main import MainStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.parser.accelerator_parser import AcceleratorParserStage
from zigzag.stages.results.reduce_stages import (
    MinimalLatencyStage,
    cheapest_then_fastest,
    fastest_then_cheapest,
    lowest_energy_delay_product,
)

EYERISS_LIKE = "zigzag/inputs/hardware/eyeriss_like.yaml"
MAPPING = [
    {
        "name": "default",
        "spatial_mapping": {},
        "spatial_mapping_hint": {},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "temporal_ordering": [],
    }
]


def best_mapping(m: int, k: int, n: int, **shortlist: Any):
    """The evaluation ZigZag keeps for an 8-bit m x k x n matmul on the Eyeriss-like array: of its fullest spatial
    mapping alone, or of 32 with the `shortlist` options."""
    gemm = {
        "id": 0,
        "name": "gemm",
        "operator_type": "Gemm",
        "equation": "O[m][n]+=I[m][k]*W[k][n]",
        "dimension_relations": [],
        "loop_dims": ["M", "K", "N"],
        "loop_sizes": [m, k, n],
        "operand_precision": {"I": 8, "W": 8, "O": 32, "O_final": 8},
        "operand_source": {"I": 0},
        "constant_operands": ["W"],
    }
    ((cme, _),) = MainStage(
        [
            MinimalLatencyStage,
            SpatialMappingGeneratorStage,
            MinimalLatencyStage,
            TemporalMappingGeneratorStage,
            CostModelStage,
        ],
        layer=LayerNodeFactory(gemm, MAPPING).create(),
        accelerator=AcceleratorParserStage.parse_accelerator(EYERISS_LIKE),
        loma_lpf_limit=6,
        temporal_mapping_type=TemporalMappingType.EVEN,
        nb_mappings_generated=32 if shortlist else 1,
        **shortlist,
    ).run()
    return cme


class Evaluation(CostModelEvaluationABC):
    """An evaluation with only the latency and energy the rankings read."""

    def __init__(self, latency: float, energy: float) -> None:
        self.latency_total2 = latency
        self.energy_total = energy


CMES = [Evaluation(latency, energy) for latency, energy in ((100, 9), (101, 5), (110, 1), (105, 2))]


@pytest.mark.parametrize(
    "ranking, order",
    [
        (fastest_then_cheapest(), [(101, 5), (100, 9), (105, 2), (110, 1)]),
        (cheapest_then_fastest(), [(110, 1), (105, 2), (101, 5), (100, 9)]),
        (lowest_energy_delay_product(), [(110, 1), (105, 2), (101, 5), (100, 9)]),
    ],
)
def test_each_ranking_puts_the_best_evaluation_first(ranking, order):
    assert [(cme.latency_total2, cme.energy_total) for cme in sorted(CMES, key=ranking(CMES))] == order


def test_the_shortlist_finds_a_mapping_faster_and_cheaper_than_the_fullest():
    fullest, shortlisted = best_mapping(64, 128, 64), best_mapping(64, 128, 64, shortlist_size=1)
    assert shortlisted.latency_total2 < fullest.latency_total2
    assert shortlisted.energy_total < fullest.energy_total


def test_a_mapping_within_the_tolerance_of_the_fastest_wins_on_energy():
    fullest, shortlisted = best_mapping(128, 128, 32), best_mapping(128, 128, 32, shortlist_size=1)
    assert fullest.latency_total2 < shortlisted.latency_total2 <= 1.02 * fullest.latency_total2
    assert shortlisted.energy_total < fullest.energy_total


def test_the_ranking_decides_against_the_fullest_mapping():
    fullest = best_mapping(128, 128, 32)
    fastest = best_mapping(128, 128, 32, shortlist_size=1, shortlist_ranking=fastest_then_cheapest(tolerance=0))
    cheapest = best_mapping(128, 128, 32, shortlist_size=1, shortlist_ranking=cheapest_then_fastest())
    assert fastest.latency_total2 <= fullest.latency_total2
    assert cheapest.energy_total < fullest.energy_total


@pytest.mark.parametrize("keeps_fullest, searched_in_full", [(True, 2), (False, 1)])
def test_only_the_shortlist_is_searched_in_full(monkeypatch, keeps_fullest, searched_in_full):
    evaluate, full = SpatialMappingGeneratorStage.evaluate, []

    def recording(stage: SpatialMappingGeneratorStage, generated_mapping, **overrides):
        if not overrides:
            full.append(generated_mapping)
        return evaluate(stage, generated_mapping, **overrides)

    monkeypatch.setattr(SpatialMappingGeneratorStage, "evaluate", recording)
    best_mapping(64, 128, 64, shortlist_size=1, shortlist_keeps_fullest=keeps_fullest)
    assert len(full) == searched_in_full


def unmappable(monkeypatch: pytest.MonkeyPatch, refuses) -> None:
    """Make the spatial mappings `refuses` accepts unmappable, as when no temporal mapping fits them."""
    evaluate = SpatialMappingGeneratorStage.evaluate

    def refusing(stage: SpatialMappingGeneratorStage, generated_mapping, **overrides):
        if refuses(generated_mapping):
            raise NoValidLoopOrderingFoundException("no fit")
        return evaluate(stage, generated_mapping, **overrides)

    monkeypatch.setattr(SpatialMappingGeneratorStage, "evaluate", refusing)


def test_a_spatial_mapping_no_temporal_mapping_fits_is_skipped(monkeypatch):
    fullest = best_mapping(64, 64, 64).layer.spatial_mapping
    unmappable(monkeypatch, lambda mapping: mapping == fullest)
    assert best_mapping(64, 64, 64, shortlist_size=1).layer.spatial_mapping != fullest


def test_a_layer_no_spatial_mapping_fits_is_refused(monkeypatch):
    unmappable(monkeypatch, lambda mapping: True)
    with pytest.raises(NoValidLoopOrderingFoundException, match="No temporal mapping fits a spatial mapping of gemm"):
        best_mapping(64, 64, 64, shortlist_size=1)
