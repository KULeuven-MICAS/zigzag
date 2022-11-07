import onnx
import pickle

from zigzag.api import get_hardware_performance
from zigzag.inputs.examples.hardware.TPU_like import accelerator
from zigzag.inputs.examples.mapping.alexnet_on_tpu_like import mapping

if __name__ == "__main__":

    # Load in your favourite onnx model
    onnx_model_path = "zigzag/inputs/examples/workload/alexnet_fp32_inferred.onnx"
    onnx_model = onnx.load(onnx_model_path, load_external_data=False)

    # Call the zigzag api, using a provided accelerator and mapping
    # energy, latency = get_hardware_performance(onnx_model, accelerator, mapping)
    energy, latency = get_hardware_performance(onnx_model, "tpu", mapping)
    print(f"Total onnx model (energy, latency) performance = ({energy:.3e}, {latency:.3e}).")
