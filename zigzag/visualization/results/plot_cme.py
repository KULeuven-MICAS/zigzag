from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from zigzag.cost_model.cost_model import CostModelEvaluation, CostModelEvaluationABC
from zigzag.datatypes import ArrayType, LayerOperand
from zigzag.hardware.architecture.memory_port import DataDirection
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.mapping.data_movement import AccessEnergy

# MPL FONT SIZES
SMALLEST_SIZE = 10
SMALLER_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 18
BIGGER_SIZE = 20
plt.rc("font", size=SMALLEST_SIZE)  # controls default text sizes # type: ignore
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title # type: ignore
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels # type: ignore
plt.rc("xtick", labelsize=SMALLER_SIZE)  # fontsize of the tick labels # type: ignore
plt.rc("ytick", labelsize=SMALLER_SIZE)  # fontsize of the tick labels # type: ignore
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize # type: ignore
plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title # type: ignore

BAR_WIDTH = 1
BAR_SPACING = 0
GROUP_SPACING = 1


def shorten_onnx_layer_name(name: str):
    """Names generated in the ONNX format are quite long (e.g. `layer1/layer1.0/conv2/Conv). This function extracts
    the most informative part"""
    try:
        return name.split("/")[-2]
    except IndexError:
        return name


def get_mem_energy_single_op(cme: CostModelEvaluation, op: LayerOperand, mem_level: int):
    return cme.mem_energy_breakdown_further[op][mem_level]


def get_energy_array(
    cmes: list[CostModelEvaluation],
    all_ops: list[LayerOperand],
    all_mems: list[MemoryInstance],
):
    """Convert the given list of cmes to a numpy array with the energy per layer, memory level, operand and data
    direction. Output shape: (cmes, all_mems, all_ops, data_directions)"""

    mem_hierarchy = cmes[0].accelerator.get_core(cmes[0].layer.core_allocation[0]).memory_hierarchy
    access_energy: dict[int, defaultdict[LayerOperand, defaultdict[MemoryInstance, AccessEnergy]]] = {
        idx: defaultdict(lambda: defaultdict(lambda: AccessEnergy(0, 0, 0, 0))) for idx in range(len(cmes))
    }

    for cme_idx, cme in enumerate(cmes):
        for operand, breakdown_per_mem in cme.mem_energy_breakdown_further.items():
            mem_op = cme.layer.memory_operand_links.layer_to_mem_op(operand)
            operand_memory_levels = mem_hierarchy.get_memory_levels(mem_op)
            for mem_idx, breakdown_this_mem in enumerate(breakdown_per_mem):
                mem = operand_memory_levels[mem_idx].memory_instance
                access_energy[cme_idx][operand][mem] += breakdown_this_mem

    mem_energy_array = np.array(
        [
            [
                [[access_energy[cme_idx][op][mem].get_single_dir_data(dir) for dir in DataDirection] for op in all_ops]
                for mem in all_mems
            ]
            for cme_idx in range(len(cmes))
        ]
    )

    mac_energy_array = np.zeros((len(cmes), 1, len(all_ops), len(DataDirection)))
    mac_energy_array[:, 0, 0, 0] = np.array([cme.mac_energy for cme in cmes])[:]

    energy_array = np.concatenate([mac_energy_array, mem_energy_array], axis=1)
    return energy_array


def get_latency_array(cmes: list[CostModelEvaluation]):
    latency_data = np.array(
        [
            [
                cme.ideal_cycle,  # Ideal computation
                cme.ideal_temporal_cycle - cme.ideal_cycle,  # Spatial stall
                cme.latency_total0 - cme.ideal_temporal_cycle,  # Temporal stall
                cme.latency_total1 - cme.latency_total0,  # Data loading
                cme.latency_total2 - cme.latency_total1,  # Data off-loading
            ]
            for cme in cmes
        ]
    )
    return latency_data


def bar_plot_cost_model_evaluations_breakdown(cmes: list[CostModelEvaluationABC], save_path: str):
    # Input-specific
    cmes_to_plot: list[CostModelEvaluation] = [cme for cme in cmes if isinstance(cme, CostModelEvaluation)]
    mem_hierarchy = cmes_to_plot[0].accelerator.get_core(cmes_to_plot[0].layer.core_allocation[0]).memory_hierarchy
    memories = [mem_level.memory_instance for mem_level in mem_hierarchy.mem_level_list]
    all_mems = sorted(memories, key=lambda x: x.size)
    all_ops = list({layer_op for cme in cmes_to_plot for layer_op in cme.layer.layer_operands})

    # Layout of data for bars
    groups = [f"{cme.layer.id}: {shorten_onnx_layer_name(cme.layer.name)}" for cme in cmes_to_plot]
    bars_energy = ["MAC"] + [mem.name for mem in all_mems]
    sections_energy = [op.name for op in all_ops]
    subsections_energy = [str(dir) for dir in DataDirection]
    sections_latency = [
        "Ideal computation",
        "Spatial stall",
        "Temporal stall",
        "Data loading",
        "Data off-loading",
    ]

    # Gather data
    energy_data = get_energy_array(cmes_to_plot, all_ops, all_mems)
    assert energy_data.shape == (
        len(groups),
        len(bars_energy),
        len(sections_energy),
        len(subsections_energy),
    )
    latency_data = get_latency_array(cmes_to_plot)
    assert latency_data.shape == (len(groups), len(sections_latency))

    # Plot config
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 8))  # type: ignore
    colors_energy = seaborn.color_palette("pastel", len(all_ops))
    colors_latency = seaborn.color_palette("muted", len(sections_latency))
    colors_latency = colors_latency[2:] + colors_latency[:2]  # Green color is at idx 3 in this palette
    color_mac = "black"
    hatches = ["////", "\\\\\\\\", "xxxx", "++++"]

    # x-axis indices
    indices = np.arange(len(groups)) * (len(bars_energy) * (BAR_WIDTH + BAR_SPACING) + GROUP_SPACING)
    indices_midpoints = indices + len(bars_energy) * (BAR_WIDTH + BAR_SPACING) / 2

    # Plot energy (MAC data at i==0)
    for i, _ in enumerate(bars_energy):
        bottom = np.zeros(len(groups))
        positions = indices + i * (BAR_WIDTH + BAR_SPACING)
        for j, section in enumerate(sections_energy):
            for k, _ in enumerate(subsections_energy):
                heights = energy_data[:, i, j, k]
                ax1.bar(  # type: ignore
                    positions,
                    heights,
                    BAR_WIDTH,
                    bottom=bottom,
                    label=f"{section}" if i == 1 and k == 0 else "",
                    color=color_mac if i == 0 else colors_energy[j],
                    edgecolor="black",
                    hatch="" if i == 0 else hatches[k],
                )
                bottom += heights

    # Total energy per layer
    bar_max_vals: ArrayType = np.max(np.sum(energy_data, axis=(2, 3)), axis=1)
    total_energy_per_layer: ArrayType = np.sum(energy_data, axis=(1, 2, 3))
    for i, _ in enumerate(groups):
        ax1.text(  # type: ignore
            indices_midpoints[i],
            1.05 * bar_max_vals[i],
            f"tot:{total_energy_per_layer[i]:.2e}",
            horizontalalignment="center",
            verticalalignment="bottom",
            weight="bold",
        )

    # Memory names as xticks
    xticks_positions = [
        indices[i] + j * (BAR_WIDTH + BAR_SPACING) for i in range(len(groups)) for j in range(len(bars_energy))
    ]
    xtick_labels = len(groups) * bars_energy
    ax1.set_xticks(xticks_positions, xtick_labels, rotation=90)  # type: ignore

    # Labels for hatches with white background
    for idx, direction in enumerate(DataDirection):
        ax1.bar(  # type: ignore
            [0],
            [0],
            width=1,
            bottom=0,
            facecolor=(1, 1, 1),
            hatch=hatches[idx],
            label=str(direction),
        )

    # Plot latency
    bottom = np.zeros(len(groups))
    for j, section in enumerate(sections_latency):
        heights = latency_data[:, j]
        ax2.bar(  # type: ignore
            indices_midpoints,
            heights,
            0.8 * len(bars_energy) * BAR_WIDTH,
            bottom=bottom,
            label=f"{section}",
            color=colors_latency[j],
            edgecolor="black",
        )
        bottom += heights

    # Total latency value above bar
    total_latency_per_layer: ArrayType = np.sum(latency_data, axis=1)
    for i in range(len(groups)):
        ax2.text(  # type: ignore
            indices_midpoints[i],
            total_latency_per_layer[i] * 1.05,
            f"tot:{total_latency_per_layer[i]:.2e}",
            horizontalalignment="center",
            verticalalignment="bottom",
            weight="bold",
        )

    # Finish details
    ax1.legend(ncol=1)  # type: ignore
    ax1.set_ylim(0, 1.1 * ax1.get_ylim()[1])  # type: ignore
    ax1.set_ylabel("Energy (pJ)", fontsize=15)  # type: ignore
    ax2.legend(ncol=1)  # type: ignore
    ax2.set_xticks(indices_midpoints, groups, rotation=0)  # type: ignore
    ax2.set_ylim(0, 1.1 * ax2.get_ylim()[1])  # type: ignore
    ax2.set_xlabel("Layers", fontsize=15)  # type: ignore
    ax2.set_ylabel("Latency (cycle)", fontsize=15)  # type: ignore

    fig.tight_layout()
    plt.savefig(save_path)  # type: ignore
    plt.close()
