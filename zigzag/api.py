import logging
from datetime import datetime
from typing import Any

from onnx import ModelProto

from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.exploit_data_locality_stages import (
    ExploitInterLayerDataLocalityStage,
    SearchInterLayerDataLocalityStage,
)
from zigzag.stages.MainStage import MainStage
from zigzag.stages.ONNXModelParserStage import ONNXModelParserStage
from zigzag.stages.reduce_stages import MinimalEDPStage, MinimalEnergyStage, MinimalLatencyStage, SumStage
from zigzag.stages.save_stages import CompleteSaveStage, PickleSaveStage, SimpleSaveStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.Stage import StageCallable
from zigzag.stages.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.VisualizationStage import VisualizationStage
from zigzag.stages.WorkloadParserStage import WorkloadParserStage
from zigzag.stages.WorkloadStage import WorkloadStage


def get_hardware_performance_zigzag(
    workload: str | ModelProto,
    accelerator: str,
    mapping: str,
    *,
    opt: str = "latency",
    dump_folder: str = f"outputs/{datetime.now()}",
    pickle_filename: str | None = None,
    lpf_limit: int = 6,
    nb_spatial_mappings_generated: int = 3,
    in_memory_compute: bool = False,
    exploit_data_locality: bool = False,
    enable_mix_spatial_mapping: bool = False,
) -> (
    tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]
    | tuple[float, float, float, float, list[tuple[CostModelEvaluationABC, Any]]]
):
    """! ZigZag API: estimates the cost of running the given workload on the given hardware architecture.
    @param workload Either a filepath to the workload ONNX or yaml file, an ONNX model.
    @param accelerator Filepath to accelerator yaml file.
    @param mapping Filepath to mapping yaml file.
    @param opt Optimization criterion: either `energy`, `latency` or `EDP`.
    @param dump_folder Folder where outputs will be saved.
    @param pickle_filename Filename of pickle dump.
    @param lpf_limit Determines the number of temporal unrollings that are evaluated.
    @param nb_spatial_mappings_generated Max nb of spatial mappings automatically generated (if not provided in
        mapping).
    @param in_memory_compute Optimizes the run for IMC architectures.
    @param exploit_data_locality Iff true, an attempt will be made to keep data in lower-level memory in between layers
    @param enable_mix_spatial_mapping Wether `mixed` spatial mappings will be generated, i.e. unrolling multiple Layer
        Dimensions in a single Operational Array Dimension.
    """
    pickle_filename = f"{dump_folder}/list_of_cmes.pickle" if pickle_filename is None else pickle_filename

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
    workload_parser_stage = (
        ONNXModelParserStage
        if isinstance(workload, ModelProto) or (workload.split(".")[-1] == "onnx")
        else WorkloadParserStage
    )

    # Add stages to keep whole layers in lower level memory instead of rewriting to DRAM, if possible
    do_exploint_inter_layer_locality = in_memory_compute or exploit_data_locality or enable_mix_spatial_mapping
    # Whether `mixed` mappings (e.g. `D1: {K:8, C:4}`) can be generated
    do_mix_spatial_mapping_generation = in_memory_compute or enable_mix_spatial_mapping

    stages = [
        # Parse the ONNX Model into the workload
        workload_parser_stage,
        # Parse the accelerator module/passthrough given accelerator
        AcceleratorParserStage,
        # Save the summed CME energy and latency to a json
        SimpleSaveStage,
        # Save all received CMEs in a list to a pickle file
        PickleSaveStage,
        # Sum up the received best CME across all layers of the workload
        SumStage,
        # Search the lowest allowed memory level per operand per layer
        SearchInterLayerDataLocalityStage if do_exploint_inter_layer_locality else None,
        # Iterate through the different layers in the workload
        WorkloadStage,
        # Save the chosen loop ordering and memory hierarchy
        VisualizationStage,
        # Remove unused memories
        ExploitInterLayerDataLocalityStage if do_exploint_inter_layer_locality else None,
        # Save each processed layer to a json
        CompleteSaveStage,
        # Reduce all CMEs, returning minimal energy/latency one
        opt_stage,
        # Generate multiple spatial mappings (SM)
        SpatialMappingGeneratorStage,
        # Reduce all CMEs, returning minimal energy/latency one
        opt_stage,
        # Generate multiple temporal mappings (TM)
        TemporalMappingGeneratorStage,
        # Evaluate generated SM and TM through cost model
        CostModelStage,
    ]

    stage_callables: list[StageCallable] = [s for s in stages if s is not None]

    # Initialize the MainStage as entry point
    mainstage = MainStage(
        list_of_callables=stage_callables,
        accelerator=accelerator,
        workload=workload,
        mapping=mapping,
        dump_folder=dump_folder,
        pickle_filename=pickle_filename,
        loma_lpf_limit=lpf_limit,
        loma_show_progress_bar=True,
        nb_mappings_generated=nb_spatial_mappings_generated,
        enable_mix_spatial_mapping_generation=do_mix_spatial_mapping_generation,
        # If we need access the same input data multiple times from the innermost memory level and the data size is
        # smaller than the memory read bw, # take into account only one-time access cost (assume the data can stay at
        # the output pins of the memory as long as it is needed).
        access_same_data_considered_as_no_access=True,
    )

    # Launch the MainStage
    cmes = mainstage.run()
    energy_total: float = cmes[0][0].energy_total
    latency_total: float = cmes[0][0].latency_total2

    if in_memory_compute:
        tclk: float = cmes[0][1][0][0].tclk
        area: float = cmes[0][1][0][0].area_total
        return energy_total, latency_total, tclk, area, cmes  # type: ignore

    return energy_total, latency_total, cmes


def get_hardware_performance_zigzag_imc(
    *args: Any,
) -> tuple[float, float, float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """Overload with type hint"""
    return get_hardware_performance_zigzag(*args, in_memory_compute=True)  # type: ignore
