from zigzag.classes.stages import *


def get_hardware_performance(onnx_model, accelerator, mapping=None, opt='latency', dump_filename_pattern="outputs/{datetime}.json", pickle_filename="outputs/list_of_cmes.pickle"):
    
    # Initialize the logger
    import logging as _logging
    _logging_level = _logging.INFO
    # _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                        format=_logging_format)

    # Sanity check on the optimization criterion
    if opt == 'energy':
        opt_stage = MinimalEnergyStage
    elif opt == 'latency':
        opt_stage = MinimalLatencyStage
    else:
        raise NotImplementedError("Optimization criterion 'opt' should be either 'energy' or 'latency'.")

    mainstage = MainStage([  # Initialize the MainStage as entry point
        ONNXModelParserStage,  # Parse the ONNX Model into the workload
        AcceleratorParserStage,  # Parse the accelerator module/passthrough given accelerator
        SimpleSaveStage,  # Save the summed CME energy and latency to a json
        PickleSaveStage,  # Save all received CMEs in a list to a pickle file
        SumStage,  # Sum up the received best CME across all layers of he workload
        WorkloadStage,  # Iterate through the different layers in the workload
        CompleteSaveStage,  # Save each processed layer to a json
        opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
        SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
        opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
        LomaStage,  # Generate multiple temporal mappings (TM)
        CostModelStage  # Evaluate generated SM and TM through cost model
    ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        onnx_model=onnx_model,  # required by ONNXModelParserStage
        mapping_path=mapping,  # required by ONNXModelParserStage
        dump_filename_pattern=dump_filename_pattern,  # output file save pattern
        pickle_filename=pickle_filename,  # filename for pickled list of cmes
        loma_lpf_limit=6  # required by LomaStage
    )

    # Launch the MainStage
    answers = mainstage.run()
    # Sanity check on the results
    assert len(answers) == 1, "Mainstage returned more than one CME."
    # Get CME from answer
    cme = answers[0][0]

    return cme.energy_total, cme.latency_total2, cme