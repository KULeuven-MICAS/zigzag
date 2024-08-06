from zigzag.parser.onnx.GemmParser import GemmParser


class MatMulParser(GemmParser):
    """! Parses an ONNX MatMul operator into a LayerNode. Exactly the same as Gemm Parser"""
