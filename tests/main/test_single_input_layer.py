from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.main import MainStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.parser.accelerator_parser import AcceleratorParserStage
from zigzag.stages.results.reduce_stages import MinimalLatencyStage

RELU = {
    "id": 0,
    "name": "relu",
    "operator_type": "Relu",
    "equation": "O[d0][d1]=I[d0][d1]",
    "dimension_relations": [],
    "loop_dims": ["D0", "D1"],
    "loop_sizes": [64, 64],
    "operand_precision": {"I": 8, "O": 8, "O_final": 8},
    "operand_source": {"I": 0},
}
MAPPING = [
    {
        "name": "default",
        "spatial_mapping": {},
        "spatial_mapping_hint": {},
        "memory_operand_links": {"O": "O", "I": "I1"},
        "temporal_ordering": [],
    }
]


def test_a_layer_with_one_input_is_costed_on_a_core_with_two_input_memories():
    """An elementwise layer reads only I1: the TPU-like core's I2 memories hold none of its operands."""
    layer = LayerNodeFactory(RELU, MAPPING).create()
    accelerator = AcceleratorParserStage.parse_accelerator("zigzag/inputs/hardware/tpu_like.yaml")
    stages = [MinimalLatencyStage, SpatialMappingGeneratorStage, MinimalLatencyStage, TemporalMappingGeneratorStage]
    (cme, _), *_ = MainStage(
        [*stages, CostModelStage],
        layer=layer,
        accelerator=accelerator,
        loma_lpf_limit=6,
        loma_show_progress_bar=False,
        temporal_mapping_type=TemporalMappingType.EVEN,
    ).run()
    assert cme.latency_total2 >= cme.ideal_cycle > 0
