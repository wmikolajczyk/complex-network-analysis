import networkx as nx
import matplotlib.pyplot as plt


def get_attributes(node_attributes, prefix):
    attributes_dict = {
        prefix + key: value
        for key, value in node_attributes
    }
    return attributes_dict


def draw_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()
