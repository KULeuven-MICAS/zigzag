import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt

from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy

def visualize_memory_hierarchy_graph(G: MemoryHierarchy):
    """
    Visualizes a memory hierarchy graph.
    """

    generations = list(nx.topological_generations(G))
    max_nodes_gen = max((len(generation) for generation in generations))
    pos = {}
    node_list = []
    node_size_list = []
    node_label_dict = {}
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

    nx.draw(G, pos=pos, node_shape='s', nodelist=node_list, node_size=node_size_list, labels=node_label_dict)
    plt.title(G.name)
    plt.show()


if __name__ == '__main__':
    import pickle
    with open('../list_of_cmes.pickle', 'rb') as handle:
        list_of_cme = pickle.load(handle)
    cme = list_of_cme[0]
    visualize_memory_hierarchy_graph(cme.accelerator.cores[0].memory_hierarchy)