import matplotlib.pyplot as plt
import networkx as nx

from zigzag.workload.DNNWorkload import DNNWorkload


def visualize_dnn_graph(graph: DNNWorkload):
    pos = {}
    pos_lb = {}
    for i_node, node in enumerate(graph.node_list):
        pos[node] = (i_node, -2 * i_node)
        pos_lb[node] = (i_node - 1.5, -2 * i_node - 2)
    plt.figure(figsize=(15, 5))  # type: ignore
    rad = 0.8
    ax = plt.gca()  # type: ignore
    edges = graph.edges()  # type: ignore
    for edge in edges:  # type: ignore
        source, target = edge  # type: ignore
        edge_rad = 0 if pos[target][0] - pos[source][0] == 1 else rad
        ax.annotate(  # type: ignore
            "",
            xy=pos[source],
            xytext=pos[target],
            arrowprops=dict(
                arrowstyle="-",
                color="black",
                connectionstyle=f"arc3,rad={edge_rad}",
                alpha=0.7,
                linewidth=5,
            ),
        )
    nx.draw_networkx_nodes(graph, pos=pos, node_size=100, node_color="black")  # type: ignore
    nx.draw_networkx_labels(graph, pos=pos_lb, font_color="black")  # type: ignore
    plt.box(False)  # type: ignore
    plt.show()  # type: ignore
