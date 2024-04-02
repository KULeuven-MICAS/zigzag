import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.utils.data as data
import torchvision
import onnx
import math
from onnx import helper, shape_inference

import json
import os
import sys
import argparse

import pickle
from zigzag.api import get_hardware_performance_zigzag
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

sys.path.append("../zigzag/inputs/examples/hardware")
sys.path.append("../zigzag/inputs/examples/mapping")


class OPTKeyConfig:

    def __init__(self, config_dict):
        self.name = config_dict["_name_or_path"].split("/")[-1]
        self.ffn_dim = config_dict["ffn_dim"]
        self.hidden_size = config_dict["hidden_size"]
        self.max_position_embeddings = config_dict["max_position_embeddings"]
        self.model_type = config_dict["model_type"]
        self.num_attention_heads = config_dict["num_attention_heads"]
        self.word_embed_proj_dim = config_dict["word_embed_proj_dim"]

        self.mapping: str
        self.accelerator: str


class GEMM(nn.Module):
    def __init__(
        self,
        model_name,
        module_name,  # Info
        mapping,
        accelerator,  # Hardware
        num_tokens,
        in_features,
        out_features,
        bias=True,  # Workload
    ):
        super(GEMM, self).__init__()
        # Info
        self.model_name = model_name
        self.module_name = module_name
        # Hardware
        self.mapping = mapping
        self.accelerator = accelerator
        # Workload
        self.num_tokens = num_tokens
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias

    def save_as_inferred_onnx(self):

        def save_as_onnx(save_path):
            self.eval()
            dummy_input = torch.randn(self.num_tokens, self.weight.shape[1])
            print(dummy_input.shape)
            folder_path = os.path.dirname(save_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.onnx.export(
                self,
                dummy_input,
                save_path,
                opset_version=12,
                input_names=["input"],
                output_names=["outputs"],
            )

        load_path = f"./outputs/{self.model_name}/{self.module_name}.onnx"
        save_path = f"./outputs/{self.model_name}/{self.module_name}_inferred.onnx"
        save_as_onnx(load_path)
        model = onnx.load(load_path)
        inferred_model = shape_inference.infer_shapes(model)
        onnx.save(inferred_model, save_path)
        # [TODO] A better hint is required
        print(
            "=========================saving inferred model as onnx========================="
        )

    def run_zigzag(self):
        workload = f"./outputs/{self.model_name}/{self.module_name}_inferred.onnx"
        dump_filename_pattern = (
            f"./outputs/{self.model_name}/Edge_TPU-{self.module_name}_?.json"
        )
        pickle_filename = f"./outputs/{self.model_name}/Edge_TPU-{self.module_name}_saved_list_of_cmes.pickle"

        energy, latency, cme = get_hardware_performance_zigzag(
            workload=workload,
            accelerator=self.accelerator,
            mapping=self.mapping,
            opt="energy",
            #    opt='latency',
            dump_filename_pattern=dump_filename_pattern,
            pickle_filename=pickle_filename,
        )

        print(f"Total network energy = {energy:.2f} pJ")
        print(f"Total network latency = {latency:.2f} cycles")

        # load in the pickled list of CMEs
        with open(pickle_filename, "rb") as fp:
            cme_for_all_layers = pickle.load(fp)
        bar_plot_cost_model_evaluations_breakdown(
            cme_for_all_layers,
            save_path=f"outputs/{self.model_name}/{self.module_name}_breakdown.png",
        )


if __name__ == "__main__":
    # Step 0: Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--config_dir", type=str, help="path of model config dirs")
    args = parser.parse_args()
    print(args)

    # Step 1: Load the model config JSON file
    config_file_path = os.path.join(args.config_dir, f"{args.model}_config.json")
    with open(config_file_path, "r") as f:
        config = json.load(f)
        config = OPTKeyConfig(config)
        config.mapping = "edge_tpu_like"
        config.accelerator = "Edge_TPU_like"
        # config.mapping = "zigzag.inputs.examples.mapping.edge_tpu_like"
        # config.accelerator = "zigzag.inputs.examples.hardware.Edge_TPU_like"

    # Step 2: Evaluate the key modules in OPT decoder layer
    #   Step 2.1: Generate the GEMM in OPT decoder layer
    #   Step 2.2: Generate the inferred GEMM modules
    #   Step 2.3: Run zigzag to evaluate the generated modules
    #   Step 2.4: Collect the results

    # test_model = GEMM(model_name=config.name,
    #                 module_name='attn.qkvo_proj',
    #                 mapping=config.mapping,
    #                 accelerator=config.accelerator,
    #                 num_tokens=config.max_position_embeddings,
    #                 in_features=config.hidden_size,
    #                 out_features=config.hidden_size,
    #             )
    # test_model.save_as_inferred_onnx()
    # test_model.run_zigzag()

    module_list = [
        # Q_proj, K_proj, V_proj, O_proj [*4]
        {
            "module_name": "attn.qkvo_proj",
            "in_features": config.hidden_size,
            "out_features": config.hidden_size,
        },
        # Q*K^T [*(num_attention_heads)]
        {
            "module_name": "attn.qk_wgt",
            "in_features": config.hidden_size // config.num_attention_heads,
            "out_features": config.max_position_embeddings,
        },
        # (Q*K^T)*V [*(num_attention_heads)]
        {
            "module_name": "attn.qkv_out",
            "in_features": config.max_position_embeddings,
            "out_features": config.hidden_size // config.num_attention_heads,
        },
        {
            "module_name": "ffn.fc1",
            "in_features": config.hidden_size,
            "out_features": config.ffn_dim,
        },
        {
            "module_name": "ffn.fc2",
            "in_features": config.ffn_dim,
            "out_features": config.hidden_size,
        },
    ]
    for module in module_list:
        print("Profiling module: ", module)
        eval_module_name = module["module_name"]
        in_features = module["in_features"]
        out_features = module["out_features"]
        eval_module = GEMM(
            model_name=config.name,
            module_name=eval_module_name,
            mapping=config.mapping,
            accelerator=config.accelerator,
            num_tokens=config.max_position_embeddings,
            in_features=in_features,
            out_features=out_features,
        )
        eval_module.save_as_inferred_onnx()
        eval_module.run_zigzag()
