from zigzag.classes.stages import *
import re

def get_hardware_performance_zigzag(workload,
                                    accelerator,
                                    mapping,
                                    opt='latency',
                                    dump_filename_pattern="outputs/{datetime}.json",
                                    pickle_filename="outputs/list_of_cmes.pickle"):
    # Initialize the logger
    import logging as _logging
    _logging_level = _logging.INFO
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                         format=_logging_format)

    # Sanity check on the optimization criterion
    if opt == 'energy':
        opt_stage = MinimalEnergyStage
    elif opt == 'latency':
        opt_stage = MinimalLatencyStage
    elif opt == 'EDP':
        opt_stage = MinimalEDPStage
    else:
        raise NotImplementedError("Optimization criterion 'opt' should be either 'energy' or 'latency' or 'EDP'.")

    # Check workload format and based on it select the correct workload parser stage
    try:
        if workload.split('.')[-1] == 'onnx':
            workload_parser_stage = ONNXModelParserStage
        else:
            workload_parser_stage = WorkloadParserStage
    except:
        workload_parser_stage = WorkloadParserStage

    mainstage = MainStage([  # Initialize the MainStage as entry point
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
        # TemporalOrderingConversionStage,  # Based on the fixed temporal mapping order, generate one temporal mapping (TM)
        CostModelStage  # Evaluate generated SM and TM through cost model
    ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=workload,  # required by workload_parser_stage
        mapping=mapping,  # required by workload_parser_stage
        dump_filename_pattern=dump_filename_pattern,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6,  # required by LomaStage
        loma_show_progress_bar=True,
    )

    # Launch the MainStage
    answers = mainstage.run()
    # Get CME from answer
    cmes = answers

    return cmes[0][0].energy_total, cmes[0][0].latency_total2, cmes


if __name__ == "__main__":
    workload = 'inputs/examples/workload/mobilenetv2.onnx'
    # workload = 'inputs.examples.workload.resnet18'
    accelerator = 'inputs.examples.hardware.TPU_like'
    mapping = 'inputs.examples.mapping.tpu_like'

    hw_name = accelerator.split(".")[-1]
    wl_name = re.split(r"/|\.", workload)[-1]
    if wl_name == 'onnx':
        wl_name = re.split(r"/|\.", workload)[-2]
    experiment_id = f"{hw_name}-{wl_name}"
    pkl_name = f'{experiment_id}-saved_list_of_cmes'

    answer = get_hardware_performance_zigzag(workload,
                                             accelerator,
                                             mapping,
                                             opt='EDP',
                                             dump_filename_pattern=f"outputs/{experiment_id}-layer_?.json",
                                             pickle_filename=f"outputs/{pkl_name}.pickle")
    # print(f'Answer = {answer}')

    # import pickle
    # path = f"outputs/{pkl_name}.pickle"
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)
    # f.close()