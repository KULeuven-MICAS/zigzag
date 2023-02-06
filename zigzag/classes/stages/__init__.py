from .CostModelStage import CostModelStage
from .DumpStage import DumpStage
from .PlotTemporalMappingsStage import PlotTemporalMappingsStage
from .SaveStage import CompleteSaveStage, SimpleSaveStage, PickleSaveStage
from .GeneralParameterIteratorStage import GeneralParameterIteratorStage
from .LomaStage import LomaStage
from .SalsaStage import SalsaStage
from .MainInputParserStages import AcceleratorParserStage, WorkloadParserStage, WorkloadAndAcceleratorParserStage
from .ONNXModelParserStage import ONNXModelParserStage
from .ReduceStages import MinimalEnergyStage, MinimalLatencyStage, MinimalEDPStage, SumStage
from .RunOptStages import CacheBeforeYieldStage, RemoveExtraInfoStage, MultiProcessingGatherStage, MultiProcessingSpawnStage, SkipIfDumpExistsStage
from .SpatialMappingConversionStage import SpatialMappingConversionStage
from .SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from .Stage import Stage, MainStage
from .TemporalOrderingConversionStage import TemporalOrderingConversionStage
from .WorkloadStage import WorkloadStage

"""
Parameter providers: these parameters are provided to substages by the following classes:
 - accelerator: AcceleratorParserStage, WorkloadAndAcceleratorParserStage, DepthFirstStage
 - workload: WorkloadParserStage, WorkloadAndAcceleratorParserStage, DepthFirstStage
 - temporal_mapping: LomaStage, TemporalMappingConversionStage
 - spatial_mapping: SpatialMappingGenerationStage, SpatialMappingConversionStage
 - layer: WorkloadStage, DepthFirstStage
 - multiprocessing_callback: MultiProcessingGatherStage
 - *:  GeneralParameterIteratorStage: can provide anything
 
Parameter consumers: these parameters are no longer provided to substages after the following classes
 - accelerator_path: AcceleratorParserStage, WorkloadAndAcceleratorParserStage
 - df_horizontal_caching: DepthFirstStage 
 - df_tilesize_x: DepthFirstStage
 - df_tilesize_y: DepthFirstStage
 - df_vertical_caching: DepthFirstStage
 - dump_filename_pattern: DumpStage
 - plot_filename_pattern: PlotTemporalMappingsStage
 - general_parameter_iterations: GeneralParameterIteratorStage
 - multiprocessing_callback: MultiProcessingSpawnStage
 - workload: DepthFirstStage, WorkloadStage
 - workload_path: WorkloadParserStage, WorkloadAndAcceleratorParserStage
 
Parameters required: these stages require the following parameters:
 - CostModelStage: accelerator, layer, spatial_mapping, temporal_mapping
 - WorkloadStage: workload
 - DepthFirstStage: accelerator, workload, df_tilesize_x, df_tilesize_y, df_horizontal_caching, df_vertical_caching
 - DumpStage: dump_filename_pattern
 - PlotTemporalMappingsStage: plot_filename_pattern
 - GeneralParameterIteratorStage: general_parameter_iterations
 - LomaStage: accelerator, layer, spatial_mapping
 - AcceleratorParserStage: accelerator_path
 - WorkloadParserStage: workload_path
 - WorkloadAndAcceleratorParserStage: workload_path, accelerator_path
 - MultiProcessingSpawnStage: multiprocessing_callback
 - SpatialMappingConversionStage: accelerator, layer
 - SpatialMappingGeneratorStage: accelerator, layer
 - TemporalOrderingConversionStage: accelerator, layer, spatial_mapping
 - SkipIfDumpExistStage: dump_filename_pattern
"""