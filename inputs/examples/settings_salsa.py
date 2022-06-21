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

'''Define the pipelines that should be used for the dse'''
from classes.stages.workload_cost_model_pipeline import WorkloadCostModelPipeline as _WorkloadCostModelPipeline
from classes.stages.workload_spatial_mapping_conversion_loma_pipeline import WorkloadSpatialMappingConversionLomaPipeline as _WorkloadSpatialMappingConversionLomaPipeline
from classes.stages.spatial_mapping_generator_salsa_pipeline import SpatialMappingGeneratorSalsaPipeline as _SpatialMappingGeneratorSalsaPipeline
pipeline = _SpatialMappingGeneratorSalsaPipeline()  # Variable name 'pipeline' should be used


'''Limit the number of temporal LPFs used for the loop ordering generation in loma'''
loma_lpf_limit = 7
visualize_loma_histogram = False

'''Salsa parameters'''
iteration_number = 1500
start_temperature = 0.05
optimization_criterion = "energy"

'''Visualization the best energy total found for different spatial mappings generated.'''
visualize_sm_energy_results = True