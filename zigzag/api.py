from onnx import ModelProto
import re
from datetime import datetime
from typing import Any

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
from zigzag.stages.LomaStage import LomaStage
from zigzag.stages.VisualizationStage import VisualizationStage
from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.stages.SearchUnusedMemoryStage import SearchUnusedMemoryStage
from zigzag.stages.RemoveUnusedMemoryStage import RemoveUnusedMemoryStage


def get_hardware_performance_zigzag(
    workload: str | dict[int, dict[str, Any]] | ModelProto,
    accelerator: str,
    mapping: str | dict[str, dict[str, Any]],
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
    lpf_limit: int = 6,
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """
    @param workload Either a filepath to the workload ONNX or yaml file, an ONNX model
    @param accelerator Filepath to accelerator yaml file
    @param mapping Filepath to mapping yaml file
    @param opt Optimization criterion: either `energy`, `latency` or `EDP`
    @param dump_folder Output folder for file dumps
    @param pickle_filename Filename of pickle dump
    @lpf_limit
    @nb_spatial_mappings_generated Max nb of spatial mappings that are automatically generated in
        SpatialMappingGeneratorStage
    """

    # Initialize the logger
    import logging as _logging

    _logging_level = _logging.INFO
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)

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
    if isinstance(workload, ModelProto) or (isinstance(workload, str) and workload.split(".")[-1] == "onnx"):
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
            LomaStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Based on the fixed temporal mapping order, generate one temporal
            # mapping (TM)
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=lpf_limit,  # required by LomaStage
        loma_show_progress_bar=True,
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
    workload: str | dict[int, dict[str, Any]] | ModelProto,
    accelerator: str,
    mapping: str | dict[str, dict[str, Any]],
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
) -> tuple[float, float, float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    # Initialize the logger
    import logging as _logging

    _logging_level = _logging.INFO
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)

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
    if isinstance(workload, ModelProto) or (isinstance(workload, str) and workload.split(".")[-1] == "onnx"):
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
            WorkloadStage,  # Iterate through the different layers in the workload
            VisualizationStage,  # Save the chosen loop ordering and memory hierarchy
            CompleteSaveStage,  # Save each processed layer to a json
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            LomaStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Based on the fixed temporal mapping order, generate one temporal
            # mapping (TM)
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by LomaStage
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
    workload: str | dict[int, dict[str, Any]] | ModelProto,
    accelerator: str,
    mapping: str | dict[str, dict[str, Any]],
    pe_array_scaling,
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    # Initialize the logger
    import logging as _logging

    _logging_level = _logging.INFO
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)

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
    if isinstance(workload, ModelProto) or (isinstance(workload, str) and workload.split(".")[-1] == "onnx"):
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
            LomaStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Based on the fixed temporal mapping order, generate one temporal
            # mapping (TM)
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by LomaStage
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


def get_hardware_performance_zigzag_without_unused_memory(
    workload: str | dict[int, dict[str, Any]] | ModelProto,
    accelerator: str,
    mapping: str | dict[str, dict[str, Any]],
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    # Initialize the logger
    import logging as _logging

    _logging_level = _logging.INFO
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)

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
    if isinstance(workload, ModelProto) or (isinstance(workload, str) and workload.split(".")[-1] == "onnx"):
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
            SearchUnusedMemoryStage,  # Search for unused memory instance
            WorkloadStage,  # Iterate through the different layers in the workload
            RemoveUnusedMemoryStage,  # Remove unused memory instance
            VisualizationStage,  # Save the chosen loop ordering and memory hierarchy
            CompleteSaveStage,  # Save each processed layer to a json
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            LomaStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Based on the fixed temporal mapping order, generate one temporal
            # mapping (TM)
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by LomaStage
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
    workload: str | dict[int, dict[str, Any]] | ModelProto,
    accelerator: str,
    mapping: str | dict[str, dict[str, Any]],
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    # Initialize the logger
    import logging as _logging

    _logging_level = _logging.INFO
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)

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
    if isinstance(workload, ModelProto) or (isinstance(workload, str) and workload.split(".")[-1] == "onnx"):
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
            SearchUnusedMemoryStage,  # Search for unused memory instance
            WorkloadStage,  # Iterate through the different layers in the workload
            RemoveUnusedMemoryStage,  # Remove unused memory instance
            VisualizationStage,  # Save the chosen loop ordering and memory hierarchy
            CompleteSaveStage,  # Save each processed layer to a json
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
            opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
            LomaStage,  # Generate multiple temporal mappings (TM)
            # TemporalOrderingConversionStage,  # Based on the fixed temporal mapping order, generate one temporal
            # mapping (TM)
            CostModelStage,  # Evaluate generated SM and TM through cost model
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_folder=dump_folder,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by LomaStage
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


if __name__ == "__main__":
    workload = "inputs/workload/mobilenetv2.onnx"
    # workload = 'inputs.examples.workload.resnet18'
    accelerator = "zigzag.inputs.examples.hardware.TPU_like"
    mapping = "zigzag.inputs.examples.mapping.tpu_like"

    hw_name = accelerator.split(".")[-1]
    wl_name = re.split(r"/|\.", workload)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload)[-2]
    experiment_id = f"{hw_name}-{wl_name}"
    pkl_name = f"{experiment_id}-saved_list_of_cmes"

    answer = get_hardware_performance_zigzag_pe_array_scaling(
        workload,
        accelerator,
        mapping,
        pe_array_scaling=2,
        opt="EDP",
        dump_filename_pattern=f"outputs/{experiment_id}-layer_?.json",
        pickle_filename=f"outputs/{pkl_name}.pickle",
    )
