import logging
from datetime import datetime

from zigzag.parser.arguments import get_arg_parser
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.ONNXModelParserStage import ONNXModelParserStage
from zigzag.stages.reduce_stages import MinimalLatencyStage
from zigzag.stages.save_stages import SimpleSaveStage
from zigzag.stages.SpatialMappingConversionStage import SpatialMappingConversionStage
from zigzag.stages.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.WorkloadStage import WorkloadStage


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    logging_level = logging.INFO
    # logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

    # Initialize the MainStage which will start execution.
    # The first argument of this init is the list of stages that will be executed in sequence.
    # The second argument of this init are the arguments required for these different stages.
    mainstage = MainStage(
        [  # Initializes the MainStage as entry point
            ONNXModelParserStage,  # Parses the ONNX Model into the workload
            AcceleratorParserStage,  # Parses the accelerator
            SimpleSaveStage,  # Saves all received CMEs information to a json
            WorkloadStage,  # Iterates through the different layers in the workload
            SpatialMappingConversionStage,  # Generates multiple spatial mappings (SM)
            MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
            TemporalMappingGeneratorStage,  # Generates multiple temporal mappings (TM)
            CostModelStage,  # Evaluates generated SM and TM through cost model
        ],
        accelerator_path=args.accelerator,  # required by AcceleratorParserStage
        onnx_model_path=args.model,  # required by ONNXModelParserStage
        mapping_path=args.mapping,  # required by ONNXModelParserStage
        dump_folder=f"outputs/{datetime.now()}",  # Output folder
        loma_lpf_limit=6,  # required by TemporalMappingGeneratorStage
        loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
    )

    # Launch the MainStage
    mainstage.run()


if __name__ == "__main__":
    main()
