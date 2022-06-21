from classes.stages import *
import argparse
parser = argparse.ArgumentParser(description="Setup zigzag-v2 inputs")
parser.add_argument('--workload', metavar='path', required=True, help='module path to workload, e.g. inputs.examples.workload1')
parser.add_argument('--accelerator', metavar='path', required=True, help='module path to the accelerator, e.g. inputs.examples.accelerator1')

args = parser.parse_args()
import logging as _logging
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

mainstage = MainStage([
    WorkloadAndAcceleratorParserStage,
    CompleteSaveStage,
    # SimpleSaveStage,
    WorkloadStage,
    SpatialMappingConversionStage,
    # SpatialMappingGeneratorStage,
    TemporalOrderingConversionStage,
    # PlotTemporalMappingsStage,
    # LomaStage,
    CostModelStage
],
    accelerator_path=args.accelerator,
    workload_path=args.workload,
    dump_filename_pattern="outputs_workshop/{datetime}.json",
    plot_filename_pattern="outputs_workshop/temporal_mappings.png"
)
mainstage.run()