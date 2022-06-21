"""
This file contains all the settings that remain fixed throughout the zigzag run.
They are parsed alongside the other inputs (e.g. accelerator, workload, ...).
See classes.io.input_config how to access these settings.

Settings starting or ending with "_" will not be parsed and remain private to avoid cluttering.
"""

'''Logging stuff'''
import logging as _logging
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

import classes.io.input_config as inputs

'''Define the pipelines that should be used for the dse'''


import argparse
parser = argparse.ArgumentParser(description="Setup zigzag-v2 inputs")
parser.add_argument('--workload', metavar='path', required=True, help='module path to workload, e.g. inputs.examples.workload1')
parser.add_argument('--accelerator', metavar='path', required=True, help='module path to the accelerator, e.g. inputs.examples.accelerator1')
args = parser.parse_args()


from classes.stages import *

pipeline = MainStage([WorkloadAndAcceleratorParserStage, DepthFirstStage, RemoveExtraInfoStage, MinimalEnergyStage, SpatialMappingConversionStage, LomaStage, CostModelStage],
                        workload_path=args.workload, accelerator_path=args.accelerator,
                        loma_lpf_limit=7,
                        visualize_loma_histogram=False,
                        visualize_sm_energy_results=False,
                        df_tilesize_x=16, df_tilesize_y=16, df_horizontal_caching=True, df_vertical_caching=True
                        )

result = pipeline.run()
print("Done")