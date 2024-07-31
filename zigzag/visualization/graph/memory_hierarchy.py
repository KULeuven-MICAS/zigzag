import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph

from zigzag.hardware.architecture.memory_level import MemoryLevel


def visualize_memory_hierarchy_graph(graph: DiGraph, save_path: str = ""):
    """
    Visualizes a memory hierarchy graph.
    """

    generations = list(nx.topological_generations(graph))  # type: ignore
    max_nodes_gen = max((len(generation) for generation in generations))
    pos = {}
    node_list = []
    node_size_list = []
    node_label_dict = {}
    node: MemoryLevel
    for gen_idx, generation in enumerate(generations):
        y = gen_idx
        node_size = (gen_idx + 1) * 2000
        for node_idx, node in enumerate(generation):
            if len(generation) == max_nodes_gen:
                x = node_idx
            else:
                x = (node_idx + 1) * (max_nodes_gen - 1) / (len(generation) + 1)
            pos[node] = (x, y)
            node_list.append(node)
            node_size_list.append(node_size)
            node_label_dict[node] = f"{node.name}\n{node.operands}\nx{node.unroll_count}"

    nx.draw(  # type: ignore
        graph,
        pos=pos,
        node_shape="s",
        nodelist=node_list,
        node_size=node_size_list,
        labels=node_label_dict,
    )
    plt.title(graph.name)  # type: ignore
    if save_path == "":
        plt.show()  # type: ignore
    else:
        plt.savefig(save_path)  # type: ignore
    plt.close()  # type: ignore


if __name__ == "__main__":
    import pickle

    with open("../list_of_cmes.pickle", "rb") as handle:
        list_of_cme = pickle.load(handle)
    cme = list_of_cme[0]
    visualize_memory_hierarchy_graph(cme.accelerator.cores[0].memory_hierarchy)
