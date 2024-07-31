import argparse


def get_arg_parser():
    # Get the onnx model, the mapping and accelerator arguments
    parser = argparse.ArgumentParser(description="Setup zigzag inputs")
    parser.add_argument(
        "--model",
        metavar="path",
        required=True,
        help="path to onnx model, e.g. inputs/my_onnx_model.onnx",
    )
    parser.add_argument(
        "--mapping",
        metavar="path",
        required=True,
        help="path to mapping file, e.g., inputs/my_mapping.yaml",
    )
    parser.add_argument(
        "--accelerator",
        metavar="path",
        required=True,
        help="module path to the accelerator, e.g. inputs.examples.accelerator1",
    )
    return parser
