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
from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.stages.SearchUnusedMemoryStage import SearchUnusedMemoryStage
from zigzag.stages.RemoveUnusedMemoryStage import RemoveUnusedMemoryStage


def get_hardware_performance_zigzag(
    workload: str | dict[int, dict[str, Any]] | ModelProto,
    accelerator: str,
    mapping: str | dict[str, dict[str, Any]],
    opt: str = "latency",
    dump_filename_pattern: str = f"outputs/{datetime.now()}.json",
    pickle_filename: str = "outputs/list_of_cmes.pickle",
    lpf_limit: int = 6,
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """
    # TODO the API should be better documented
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
        dump_filename_pattern=dump_filename_pattern,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=lpf_limit,  # required by LomaStage
        loma_show_progress_bar=True,
        enable_weight_diagonal_mapping=False,
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
    dump_filename_pattern: str = "outputs/layer_?.json",
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
            SearchUnusedMemoryStage,  # Detect unnecessary memory instances
            WorkloadStage,  # Iterate through the different layers in the workload
            RemoveUnusedMemoryStage,  # Remove unnecessary memory instances
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
        dump_filename_pattern=dump_filename_pattern,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by LomaStage
        enable_mix_spatial_mapping_generation=True,  # enable auto-generation of mix spatial mapping
        maximize_hardware_utilization=True,  # only evaluate spatial mapping with top2 utilization (fast simulation)
        enable_weight_diagonal_mapping=True,  # required by SpatialMappingGeneratorStage
        loma_show_progress_bar=True,
        # If we need access the same input data multiple times from the innermost memory level and the data size is
        # smaller than the memory read bw,
        # take into account only one-time access cost (assume the data can stay at the output pins of the memory as
        # long as it is needed).
        # By default, if the parameter is not defined, it will be set as False internally.
        access_same_data_considered_as_no_access=True,
    )

    # Launch the MainStage
    cmes = mainstage.run()

    return (
        cmes[0][0].energy_total,
        cmes[0][0].latency_total2,
        cmes[0][0].tclk,
        cmes[0][0].area_total,
        cmes,
    )


def get_hardware_performance_zigzag_pe_array_scaling(
    workload: str | dict[int, dict[str, Any]] | ModelProto,
    accelerator: str,
    mapping: str | dict[str, dict[str, Any]],
    pe_array_scaling,
    opt: str = "latency",
    dump_filename_pattern: str = "outputs/{datetime}.json",
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
        dump_filename_pattern=dump_filename_pattern,  # output file save pattern
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
    dump_filename_pattern: str = "outputs/{datetime}.json",
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
        dump_filename_pattern=dump_filename_pattern,  # output file save pattern
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
    dump_filename_pattern: str = "outputs/{datetime}.json",
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
        dump_filename_pattern=dump_filename_pattern,  # output file save pattern
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
    workload = "zigzag/inputs/examples/workload/mobilenetv2.onnx"
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
