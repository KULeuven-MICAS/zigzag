import matplotlib.pyplot as plt
import networkx as nx


def visualize_dnn_graph(G):
    pos = {}
    pos_lb = {}
    for i_node, node in enumerate(G.nodes()):
        pos[node] = (i_node, -2*i_node)
        pos_lb[node] = (i_node-1.5, -2*i_node-2)
    plt.figure(figsize=(15, 5))
    rad = 0.8
    ax = plt.gca()
    edges = G.edges()
    for edge in edges:
        source, target = edge
        edge_rad = 0 if pos[target][0] - pos[source][0] == 1 else rad
        ax.annotate("",
                    xy=pos[source],
                    xytext=pos[target],
                    arrowprops=dict(arrowstyle="-", color="black",
                                    connectionstyle=f"arc3,rad={edge_rad}",
                                    alpha=0.7,
                                    linewidth=5))
    nx.draw_networkx_nodes(G, pos=pos, node_size=100, node_color='black')
    nx.draw_networkx_labels(G, pos=pos_lb, font_color='black')
    plt.box(False)
    plt.show()


if __name__ == '__main__':
    import zigzag.classes.stages.MainInputParserStages as MainInputParserStages
    workload = 'zigzag.inputs.examples.workload.resnet18'
    parsed_workload = MainInputParserStages.parse_workload_from_path_or_from_module(workload)
    visualize_dnn_graph(parsed_workload)