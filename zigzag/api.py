from onnx import ModelProto
from datetime import datetime
from typing import Any
import logging

# from zigzag.stages.TemporalOrderingConversionStage import TemporalOrderingConversionStage
from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.ONNXModelParserStage import ONNXModelParserStage
from zigzag.stages.PEArrayScalingStage import PEArrayScalingStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.WorkloadStage import WorkloadStage
from zigzag.stages.WorkloadParserStage import WorkloadParserStage
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.reduce_stages import MinimalEDPStage, MinimalEnergyStage, MinimalLatencyStage, SumStage
from zigzag.stages.save_stages import CompleteSaveStage, PickleSaveStage, SimpleSaveStage
from zigzag.stages.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.VisualizationStage import VisualizationStage
from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.stages.exploit_data_locality_stages import (
    SearchInterLayerDataLocalityStage,
    ExploitInterLayerDataLocalityStage,
)


def get_hardware_performance_zigzag(
    workload: str | ModelProto,
    accelerator: str,
    mapping: str,
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
    lpf_limit: int = 6,
    nb_spatial_mappings_generated: int = 3,
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """! Function of deriving the accelerator cost (both digital and in-memory-computing cores are supported)
    @param workload Either a filepath to the workload ONNX or yaml file, an ONNX model
    @param accelerator Filepath to accelerator yaml file
    @param mapping Filepath to mapping yaml file
    @param opt Optimization criterion: either `energy`, `latency` or `EDP`
    @param dump_filename_pattern Filename pattern for file dumps
    @param pickle_filename Filename of pickle dump
    @lpf_limit
    @nb_spatial_mappings_generated Max nb of spatial mappings that are automatically generated in
        SpatialMappingGeneratorStage
    """

    # Initialize the logger
    logging_level = logging.INFO
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

    match opt:
        case "energy":
            opt_stage = MinimalEnergyStage
        case "latency":
            opt_stage = MinimalLatencyStage
        case "EDP":
            opt_stage = MinimalEDPStage
        case _:
            raise NotImplementedError("Optimization criterion 'opt' should be either 'energy' or 'latency' or 'EDP'.")

    # Check workload format and based on it select the correct workload parser stage
    if isinstance(workload, ModelProto) or (workload.split(".")[-1] == "onnx"):
        workload_parser_stage = ONNXModelParserStage
    else:
        workload_parser_stage = WorkloadParserStage

    mainstage = MainStage(
        [  # Initialize the MainStage as entry point
            workload_parser_stage,  # Parse the ONNX Model into the workload
            AcceleratorParserStage,  # Parse the accelerator module/passthrough given accelerator
            SimpleSaveStage,  # Save the summed CME energy and latency to a json
            PickleSaveStage,  # Save all received CMEs in a list to a pickle file
            SumStage,  # Sum up the received best CME across all layers of the workload
            WorkloadStage,  # Iterate through the different layers in the workload
            VisualizationStage,  # Save the chosen loop ordering and memory hierarchy
            CompleteSaveStage,  # Save each processed layer to a json
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            TemporalMappingGeneratorStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Parse user-defined fixed temporal mapping order
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=lpf_limit,  # required by TemporalMappingGeneratorStage
        loma_show_progress_bar=True,
        # Max nb of spatial mappings that are automatically generated in SpatialMappingGeneratorStage
        nb_mappings_generated=nb_spatial_mappings_generated,
        # Whether `mixed` mappings (e.g. `D1: {K:8, C:4}`) can be generated
        enable_mix_spatial_mapping_generation=False,
        # If we need access the same input data multiple times from the innermost memory level and the data size is
        # smaller than the memory read bw,
        # take into account only one-time access cost (assume the data can stay at the output pins of the memory as
        # long as it is needed).
        # By default, if the parameter is not defined, it will be set as False internally.
        access_same_data_considered_as_no_access=True,
    )

    # Launch the MainStage
    cmes = mainstage.run()

    return cmes[0][0].energy_total, cmes[0][0].latency_total2, cmes


def get_hardware_performance_zigzag_imc(
    workload: str | ModelProto,
    accelerator: str,
    mapping: str,
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
) -> tuple[float, float, float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """! Function of deriving cost of solely in-memory computing accelerators (tclk and area will be returned)"""

    # Initialize the logger
    logging_level = logging.INFO
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

    match opt:
        case "energy":
            opt_stage = MinimalEnergyStage
        case "latency":
            opt_stage = MinimalLatencyStage
        case "EDP":
            opt_stage = MinimalEDPStage
        case _:
            raise NotImplementedError("Optimization criterion 'opt' should be either 'energy' or 'latency' or 'EDP'.")

    # Check workload format and based on it select the correct workload parser stage
    if isinstance(workload, ModelProto) or workload.split(".")[-1] == "onnx":
        workload_parser_stage = ONNXModelParserStage
    else:
        workload_parser_stage = WorkloadParserStage

    mainstage = MainStage(
        [  # Initialize the MainStage as entry point
            workload_parser_stage,  # Parse the ONNX Model into the workload
            AcceleratorParserStage,  # Parse the accelerator module/passthrough given accelerator
            CompleteSaveStage,  # Save the summed CME energy and latency to a json
            PickleSaveStage,  # Save all received CMEs in a list to a pickle file
            SumStage,  # Sum up the received best CME across all layers of the workload
            SearchInterLayerDataLocalityStage,  # Search the lowest allowed memory level per operand per layer
            WorkloadStage,  # Iterate through the different layers in the workload
            VisualizationStage,  # Save the chosen loop ordering and memory hierarchy
            ExploitInterLayerDataLocalityStage,  # Remove unused memories
            CompleteSaveStage,  # Save each processed layer to a json
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            TemporalMappingGeneratorStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Parse user-defined fixed temporal mapping order
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by TemporalMappingGeneratorStage
        loma_show_progress_bar=True,
        enable_mix_spatial_mapping_generation=True,
        nb_mappings_generated=3,
        # If we need access the same input data multiple times from the innermost memory level and the data size is
        # smaller than the memory read bw,
        # take into account only one-time access cost (assume the data can stay at the output pins of the memory as
        # long as it is needed).
        # By default, if the parameter is not defined, it will be set as False internally.
        access_same_data_considered_as_no_access=True,
    )

    # Launch the MainStage
    cmes = mainstage.run()
    energy_total: float = cmes[0][0].energy_total
    latency_total: float = cmes[0][0].latency_total2
    tclk: float = cmes[0][1][0][0].tclk
    area: float = cmes[0][1][0][0].area_total

    return energy_total, latency_total, tclk, area, cmes


def get_hardware_performance_zigzag_pe_array_scaling(
    workload: str | ModelProto,
    accelerator: str,
    mapping: str,
    pe_array_scaling: int,
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """! Function of deriving accelerator cost where the pe array size need to be scaled"""

    # Initialize the logger
    logging_level = logging.INFO
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

    match opt:
        case "energy":
            opt_stage = MinimalEnergyStage
        case "latency":
            opt_stage = MinimalLatencyStage
        case "EDP":
            opt_stage = MinimalEDPStage
        case _:
            raise NotImplementedError("Optimization criterion 'opt' should be either 'energy' or 'latency' or 'EDP'.")

    # Check workload format and based on it select the correct workload parser stage
    if isinstance(workload, ModelProto) or workload.split(".")[-1] == "onnx":
        workload_parser_stage = ONNXModelParserStage
    else:
        workload_parser_stage = WorkloadParserStage

    mainstage = MainStage(
        [  # Initialize the MainStage as entry point
            workload_parser_stage,  # Parse the ONNX Model into the workload
            AcceleratorParserStage,  # Parse the accelerator module/passthrough given accelerator
            PEArrayScalingStage,  # Scale the PE array of the given accelerator
            SimpleSaveStage,  # Save the summed CME energy and latency to a json
            PickleSaveStage,  # Save all received CMEs in a list to a pickle file
            SumStage,  # Sum up the received best CME across all layers of the workload
            WorkloadStage,  # Iterate through the different layers in the workload
            VisualizationStage,  # Save the chosen loop ordering and memory hierarchy
            CompleteSaveStage,  # Save each processed layer to a json
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            TemporalMappingGeneratorStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Parse user-defined fixed temporal mapping order
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by TemporalMappingGeneratorStage
        loma_show_progress_bar=True,
        # If we need access the same input data multiple times from the innermost memory level and the data size is
        # smaller than the memory read bw,
        # take into account only one-time access cost (assume the data can stay at the output pins of the memory as long
        # as it is needed).
        # By default, if the parameter is not defined, it will be set as False internally.
        access_same_data_considered_as_no_access=True,
        pe_array_scaling=pe_array_scaling,
    )

    # Launch the MainStage
    answers = mainstage.run()
    # Get CME from answer
    cmes = answers

    return cmes[0][0].energy_total, cmes[0][0].latency_total2, cmes


def get_hardware_performance_zigzag_with_exploit_data_locality(
    workload: str | ModelProto,
    accelerator: str,
    mapping: str,
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """! Function of deriving cost when output of intermediate layers is kept in memory levels as low as possible"""

    # Initialize the logger
    logging_level = logging.INFO
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

    match opt:
        case "energy":
            opt_stage = MinimalEnergyStage
        case "latency":
            opt_stage = MinimalLatencyStage
        case "EDP":
            opt_stage = MinimalEDPStage
        case _:
            raise NotImplementedError("Optimization criterion 'opt' should be either 'energy' or 'latency' or 'EDP'.")

    # Check workload format and based on it select the correct workload parser stage
    if isinstance(workload, ModelProto) or workload.split(".")[-1] == "onnx":
        workload_parser_stage = ONNXModelParserStage
    else:
        workload_parser_stage = WorkloadParserStage

    mainstage = MainStage(
        [  # Initialize the MainStage as entry point
            workload_parser_stage,  # Parse the ONNX Model into the workload
            AcceleratorParserStage,  # Parse the accelerator module/passthrough given accelerator
            SimpleSaveStage,  # Save the summed CME energy and latency to a json
            PickleSaveStage,  # Save all received CMEs in a list to a pickle file
            SumStage,  # Sum up the received best CME across all layers of the workload
            SearchInterLayerDataLocalityStage,  # Search for unused memory instance
            WorkloadStage,  # Iterate through the different layers in the workload
            ExploitInterLayerDataLocalityStage,  # Remove unused memory instance
            VisualizationStage,  # Save the chosen loop ordering and memory hierarchy
            CompleteSaveStage,  # Save each processed layer to a json
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            TemporalMappingGeneratorStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Parse user-defined fixed temporal mapping order
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by TemporalMappingGeneratorStage
        loma_show_progress_bar=True,
        # If we need access the same input data multiple times from the innermost memory level and the data size is
        # smaller than the memory read bw,
        # take into account only one-time access cost (assume the data can stay at the output pins of the memory as long
        # as it is needed).
        # By default, if the parameter is not defined, it will be set as False internally.
        access_same_data_considered_as_no_access=True,
    )

    # Launch the MainStage
    answers = mainstage.run()
    # Get CME from answer
    cmes = answers

    return cmes[0][0].energy_total, cmes[0][0].latency_total2, cmes


def get_hardware_performance_zigzag_with_mix_spatial_mapping(
    workload: str | ModelProto,
    accelerator: str,
    mapping: str,
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """! Function of deriving accelerator cost when a mixed spatial mapping is required"""

    # Initialize the logger
    logging_level = logging.INFO
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)

    match opt:
        case "energy":
            opt_stage = MinimalEnergyStage
        case "latency":
            opt_stage = MinimalLatencyStage
        case "EDP":
            opt_stage = MinimalEDPStage
        case _:
            raise NotImplementedError("Optimization criterion 'opt' should be either 'energy' or 'latency' or 'EDP'.")

    # Check workload format and based on it select the correct workload parser stage
    if isinstance(workload, ModelProto) or workload.split(".")[-1] == "onnx":
        workload_parser_stage = ONNXModelParserStage
    else:
        workload_parser_stage = WorkloadParserStage

    mainstage = MainStage(
        [  # Initialize the MainStage as entry point
            workload_parser_stage,  # Parse the ONNX Model into the workload
            AcceleratorParserStage,  # Parse the accelerator module/passthrough given accelerator
            SimpleSaveStage,  # Save the summed CME energy and latency to a json
            PickleSaveStage,  # Save all received CMEs in a list to a pickle file
            SumStage,  # Sum up the received best CME across all layers of the workload
            SearchInterLayerDataLocalityStage,  # Search for unused memory instance
            WorkloadStage,  # Iterate through the different layers in the workload
            ExploitInterLayerDataLocalityStage,  # Remove unused memory instance
            VisualizationStage,  # Save the chosen loop ordering and memory hierarchy
            CompleteSaveStage,  # Save each processed layer to a json
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            TemporalMappingGeneratorStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Parse user-defined fixed temporal mapping order
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by TemporalMappingGeneratorStage
        loma_show_progress_bar=True,
        # If we need access the same input data multiple times from the innermost memory level and the data size is
        # smaller than the memory read bw,
        # take into account only one-time access cost (assume the data can stay at the output pins of the memory as long
        # as it is needed).
        # By default, if the parameter is not defined, it will be set as False internally.
        access_same_data_considered_as_no_access=True,
        enable_mix_spatial_mapping_generation=True,  # enable auto-generation of mix spatial mapping
    )

    # Launch the MainStage
    answers = mainstage.run()
    # Get CME from answer
    cmes = answers

    return cmes[0][0].energy_total, cmes[0][0].latency_total2, cmes
