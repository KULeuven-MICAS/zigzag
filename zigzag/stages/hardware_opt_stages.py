import logging
from itertools import product
from typing import Any

import matplotlib.pyplot as plt
import yaml

from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.mapping.spatial_mapping import MappingSingleOADim, SpatialMapping
from zigzag.stages.stage import Stage, StageCallable
from zigzag.workload.dnn_workload import DNNWorkload
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class HardwareOptimizerStage(Stage):
    """! Class that iterates through different memories for the l1 memory of the gemm_l1_l3 hardware.
    The best one is kept based on latency.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        accelerator: str,
        workload: DNNWorkload,
        **kwargs: Any,
    ):
        """
        Initialize the stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        self.output_path = self.kwargs.get("dump_folder", "./")

        # Define different l1 memory sizes and gemmx array sizes to try
        self.l1_memory_sizes = [32, 64, 128, 256, 512]  # KiB
        self.l1_memory_sizes_read_costs = {
            32: 95,
            64: 104,
            128: 118,
            256: 132,
            512: 147,
        }  # pJ/64B
        # Define different gemmx array sizes to try
        self.gemmx_array_sizes = [8, 16]  # Combinations will be tried for every dimension (array is 3D)
        self.gemmx_array_size_combinations = list(product(self.gemmx_array_sizes, repeat=3))

    def run(self):
        """! Run the compare stage by comparing a new cost model output with the current best found result."""
        sub_list_of_callables = self.list_of_callables[1:]
        all_cmes: list[CostModelEvaluationABC] = []
        best_cme: CostModelEvaluationABC | None = None
        for accelerator_path, workload in self._accelerator_and_workload_iterator():
            logger.info(f"Evaluating hardware configuration: {accelerator_path}")
            substage = self.list_of_callables[0](
                sub_list_of_callables,
                accelerator=accelerator_path,
                workload=workload,
                **self.kwargs,
            )

            for cme, _ in substage.run():
                assert isinstance(cme, CostModelEvaluationABC)
                if (
                    best_cme is None
                    or cme.latency_total2 < best_cme.latency_total2
                    or (cme.latency_total2 == best_cme.latency_total2 and cme.energy_total < best_cme.energy_total)
                ):
                    best_cme = cme
                all_cmes.append(cme)

        self._plot_cme_distributions(all_cmes, best_cme)
        assert best_cme is not None
        yield best_cme, [(best_cme, None)]

    def _accelerator_and_workload_iterator(self) -> list[tuple[str, DNNWorkload]]:
        """Iterate through different hardware configurations and update the workload spatial unrolling."""
        # First, assert that the accelerator yaml name is gemm_l1_l3
        assert "gemm_l1_l3" in self.accelerator, "HardwareOptimizerStage only supports gemm_l1_l3 accelerator."
        # Load in the yaml hardware
        with open(self.accelerator, "r", encoding="UTF-8") as f:
            accel_yaml = yaml.safe_load(f)
        # Next, assert that the workload only has one layer of type gemm
        assert len(self.workload.layer_node_list) == 1, "HardwareOptimizerStage only supports single-layer for now."
        layer = self.workload.layer_node_list[0]
        assert layer.type == "Gemm", "HardwareOptimizerStage only supports single Gemm layer for now."
        original_spatial_mapping = layer.spatial_mapping
        i = 0
        for l1_size in self.l1_memory_sizes:
            for array_size in self.gemmx_array_size_combinations:
                assert len(array_size) == 3
                accel_yaml_modified = accel_yaml.copy()
                accel_yaml_modified = self._update_l1_size_in_yaml(accel_yaml_modified, l1_size)
                accel_yaml_modified = self._update_gemmx_array_size_in_yaml(accel_yaml_modified, array_size)
                accel_yaml_modified = self._update_bandwidths_in_yaml(accel_yaml_modified, array_size)
                layer.spatial_mapping = self._update_workload_spatial_mapping(
                    original_spatial_mapping, array_size, layer
                )
                modified_accel_path = f"{self.output_path}/gemm_l1_l3_{i}.yaml"
                with open(modified_accel_path, "w", encoding="UTF-8") as f:
                    yaml.dump(accel_yaml_modified, f, sort_keys=False)
                i += 1
                yield modified_accel_path, self.workload

    def _update_l1_size_in_yaml(self, accel_yaml: dict[str, Any], l1_size: int) -> dict[str, Any]:
        """Update the l1 memory size in the accelerator yaml dictionary."""
        accel_yaml["memories"]["l1"]["size"] = int(
            l1_size * 1024 * 8 * 0.8
        )  # Convert KiB to bits and subtract 20% for snitch stack
        read_cost = self.l1_memory_sizes_read_costs[l1_size]
        write_cost = 1.2 * read_cost
        accel_yaml["memories"]["l1"]["r_cost"] = read_cost
        accel_yaml["memories"]["l1"]["w_cost"] = write_cost
        return accel_yaml

    def _update_gemmx_array_size_in_yaml(
        self, accel_yaml: dict[str, Any], array_size: tuple[int, int, int]
    ) -> dict[str, Any]:
        """Update the gemmx array sizes in the accelerator yaml dictionary."""
        accel_yaml["operational_array"]["sizes"] = list(array_size)
        return accel_yaml

    def _update_bandwidths_in_yaml(
        self, accel_yaml: dict[str, Any], array_size: tuple[int, int, int]
    ) -> dict[str, Any]:
        """Update the bandwidths in the accelerator yaml dictionary based on the gemmx array sizes."""
        # l1 rw_port_1 bandwidth is D1.size * D2.size * 8bit/I
        rw_1_bw = array_size[0] * array_size[1] * 8
        rw_port_1 = accel_yaml["memories"]["l1"]["ports"][0]
        assert rw_port_1["name"] == "rw_port_1", "Expected rw_port_1 to be the first port of l1 memory."
        rw_port_1["bandwidth_max"] = rw_1_bw

        # l1 rw_port_2 bandwidth is D2.size * D3.size * 8bit/W
        rw_2_bw = array_size[1] * array_size[2] * 8
        rw_port_2 = accel_yaml["memories"]["l1"]["ports"][1]
        assert rw_port_2["name"] == "rw_port_2", "Expected rw_port_2 to be the second port of l1 memory."
        rw_port_2["bandwidth_max"] = rw_2_bw

        # l1 rw_port_3 bandwidth is D1.size * D3.size * 32bit/O
        rw_3_bw = array_size[0] * array_size[2] * 32
        rw_port_3 = accel_yaml["memories"]["l1"]["ports"][2]
        assert rw_port_3["name"] == "rw_port_3", "Expected rw_port_3 to be the third port of l1 memory."
        rw_port_3["bandwidth_max"] = rw_3_bw

        return accel_yaml

    def _update_workload_spatial_mapping(
        self, original_spatial_mapping: SpatialMapping, array_size: tuple[int, int, int], layer: LayerNode
    ) -> SpatialMapping:
        """Update the workload spatial mapping unrolling based on the gemmx array sizes."""
        new_spatial_mapping = original_spatial_mapping.copy()
        # Update unrolling for D1, D2, D3 based on array sizes
        for oa_dim, unrolling in original_spatial_mapping.items():
            oa_dim_size = array_size[int(oa_dim.name[1]) - 1]  # D1 -> index 0, D2 -> index 1, D3 -> index 2
            layer_dim = next(iter(unrolling.get_data()))
            layer_dim_size = layer.layer_dim_sizes[layer_dim]
            _, rem = divmod(layer_dim_size, oa_dim_size)
            if rem != 0:
                raise ValueError(
                    f"Array size {oa_dim_size} does not evenly divide layer dimension size {layer_dim_size} for {layer_dim}."
                )
            new_spatial_mapping[oa_dim] = MappingSingleOADim({layer_dim: oa_dim_size})
        return new_spatial_mapping

    def _plot_cme_distributions(
        self, all_cmes: list[CostModelEvaluationABC], best_cme: CostModelEvaluationABC | None = None
    ) -> None:
        """Plot boxplots of CME latencies and energies for all evaluated configurations."""
        latencies = [cme.latency_total2 for cme in all_cmes]
        energies = [cme.energy_total for cme in all_cmes]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.boxplot(latencies, vert=True)
        if best_cme is not None:
            plt.plot(1, best_cme.latency_total2, "r+", markersize=15, markeredgewidth=2)
        plt.title("CME Latency Distribution")
        plt.ylabel("Latency (cycles)")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.xticks([])

        plt.subplot(1, 2, 2)
        plt.boxplot(energies, vert=True)
        if best_cme is not None:
            plt.plot(1, best_cme.energy_total, "r+", markersize=15, markeredgewidth=2)
        plt.title("CME Energy Distribution")
        plt.ylabel("Energy (pJ)")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.xticks([])

        plt.tight_layout()
        plot_path = f"{self.output_path}/cme_distributions.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"CME distributions saved to {plot_path}")
