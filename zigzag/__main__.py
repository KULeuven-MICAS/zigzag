from datetime import datetime
from zigzag.stages import *

from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.ONNXModelParserStage import ONNXModelParserStage
from zigzag.stages.SpatialMappingConversionStage import SpatialMappingConversionStage
from zigzag.stages.WorkloadStage import WorkloadStage
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.reduce_stages import MinimalLatencyStage
from zigzag.stages.save_stages import SimpleSaveStage
from zigzag.stages.LomaStage import LomaStage
from zigzag.parser.arguments import get_arg_parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Initialize the logger
    import logging as _logging

    _logging_level = _logging.INFO
    # _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)

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
            LomaStage,  # Generates multiple temporal mappings (TM)
            CostModelStage,  # Evaluates generated SM and TM through cost model
        ],
        accelerator_path=args.accelerator,  # required by AcceleratorParserStage
        onnx_model_path=args.model,  # required by ONNXModelParserStage
        mapping_path=args.mapping,  # required by ONNXModelParserStage
        dump_folder=f"outputs/{datetime.now()}",  # Output folder
        loma_lpf_limit=6,  # required by LomaStage
        loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
    )

    # Launch the MainStage
    mainstage.run()


if __name__ == "__main__":
    main()
