"""
=====================================================================
Title:        main_onnx_salsa.py
Description:

Date:        02.01.2023

=====================================================================

Copyright (C) 2020 ETH Zurich and University of Bologna.

Author: Victor Jung, ETH Zurich

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the License); you may
not use this file except in compliance with the License.
You may obtain a copy of the License at

www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import re

from zigzag.parser.arguments import get_arg_parser
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.ONNXModelParserStage import ONNXModelParserStage
from zigzag.stages.reduce_stages import MinimalLatencyStage
from zigzag.stages.SalsaStage import SalsaStage
from zigzag.stages.save_stages import SimpleSaveStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.WorkloadStage import WorkloadStage

parser = get_arg_parser()
args = parser.parse_args()

# Initialize the logger

logging_level = logging.INFO
logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format)

hw_name = args.accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", args.model)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", args.model)[-2]
experiment_id = f"{hw_name}-{wl_name}"
pkl_name = f"{experiment_id}-saved_list_of_cmes"

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        ONNXModelParserStage,  # Parses the ONNX Model into the workload
        AcceleratorParserStage,  # Parses the accelerator
        SimpleSaveStage,  # Saves all received CMEs information to a json
        WorkloadStage,  # Iterates through the different layers in the workload
        SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
        MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
        SalsaStage,  # Find pseudo-optimal temporal mapping
        CostModelStage,  # Evaluates generated SM and TM through cost model
    ],
    accelerator=args.accelerator,  # required by AcceleratorParserStage
    workload=args.model,  # required by ONNXModelParserStage
    mapping=args.mapping,  # required by ONNXModelParserStage
    dump_folder=f"outputs/{experiment_id}",  # Output folder
    loma_lpf_limit=6,  # required by TemporalMappingGeneratorStage
    loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
    salsa_iteration_number=1000,
    salsa_start_temperature=0.05,
    salsa_opt_criterion="latency",
    salsa_number_of_core=8,
)

# Launch the MainStage
mainstage.run()
