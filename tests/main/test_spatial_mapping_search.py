import itertools

from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.parser.accelerator_parser import AcceleratorParserStage

CONV = {
    "id": 0,
    "name": "conv",
    "operator_type": "Conv",
    "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
    "dimension_relations": ["ix=1*ox+1*fx", "iy=1*oy+1*fy"],
    "loop_dims": ["B", "K", "C", "OY", "OX", "FY", "FX"],
    "loop_sizes": [1, 64, 32, 28, 28, 3, 3],
    "operand_precision": {"I": 8, "W": 8, "O": 16, "O_final": 8},
    "operand_source": {"I": 0},
    "constant_operands": ["W"],
}
MAPPING = [
    {
        "name": "default",
        "spatial_mapping": {},
        "spatial_mapping_hint": {},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "temporal_ordering": [],
    }
]


def test_the_search_returns_the_mappings_sorting_every_combination_gives():
    """With mixed unrolling and no hints, the search keeps exactly the mappings that sorting every valid combination
    keeps, in the same order."""
    layer = LayerNodeFactory(CONV, MAPPING).create()
    accelerator = AcceleratorParserStage.parse_accelerator("zigzag/inputs/hardware/tpu_like.yaml")
    stage = SpatialMappingGeneratorStage(
        [CostModelStage],
        accelerator=accelerator,
        layer=layer,
        enable_mix_spatial_mapping_generation=True,
        nb_mappings_generated=5,
    )
    max_unrollings = stage.get_max_unrolling()
    template = stage.provided_mapping.copy()
    template.initialize_oa_dims(stage.oa_dim_sizes)
    oa_dims = list(stage.oa_dim_sizes)
    options = [
        list(
            stage.generate_spatial_mapping_single_oa_dim(
                stage.spatial_mapping_hint[d], max_unrollings[d], stage.oa_dim_sizes[d]
            )
        )
        for d in oa_dims
    ]

    every = []
    for combination in itertools.product(*options):
        candidate = template.copy()
        for oa_dim, mapping in zip(oa_dims, combination):
            candidate[oa_dim] = mapping
        if candidate.is_valid(max_unrollings, stage.layer_dim_sizes.data):
            every.append(candidate)
    expected = sorted(every, key=lambda x: x.get_performance_indicator(), reverse=True)[:5]

    found = stage.best_candidates(template, oa_dims, (iter(o) for o in options), max_unrollings)
    assert [str(m) for m in found] == [str(m) for m in expected]
