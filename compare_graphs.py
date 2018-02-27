# -*- coding: utf-8
import networkx as nx

from collections import OrderedDict

from scipy import stats


def compare(graph1, graph2):
    """
    Compare two graphs

    Args:
        graph1: First graph to compare
        graph2: Second graph to compare

    Returns:
        A dict with calculated diffrence measures
    """
    result = OrderedDict()
    # KS test for graph measurements distributions
    graph1_degree_centrality = list(nx.degree_centrality(graph1).values())
    graph2_degree_centrality = list(nx.degree_centrality(graph2).values())
    result['degree_centrality'] = stats.ks_2samp(graph1_degree_centrality,
                                                 graph2_degree_centrality)

    graph1_closeness_centrality = list(
        nx.closeness_centrality(graph1).values())
    graph2_closeness_centrality = list(
        nx.closeness_centrality(graph2).values())
    result['closeness_centrality'] = stats.ks_2samp(
        graph1_closeness_centrality, graph2_closeness_centrality)

    graph1_betweenness_centrality = list(
        nx.betweenness_centrality(graph1).values())
    graph2_betweenness_centrality = list(
        nx.betweenness_centrality(graph2).values())
    result['betweenness_centrality'] = stats.ks_2samp(
        graph1_betweenness_centrality, graph2_betweenness_centrality)

    graph1_pagerank = list(nx.pagerank(graph1).values())
    graph2_pagerank = list(nx.pagerank(graph2).values())
    result['pagerank'] = stats.ks_2samp(graph1_pagerank, graph2_pagerank)

    # absolute value of global graph measurements subtraction
    result['average_shortest_path_length'] = abs(
        nx.average_shortest_path_length(
            graph1) - nx.average_shortest_path_length(graph2))
    result['diameter'] = abs(nx.diameter(graph1) - nx.diameter(graph2))
    graph1_max_degree = max([v for k, v in graph1.degree])
    graph1_degree_deltas = [max_degree - v for k, v in graph1.degree]
    graph1_degree_centralization = sum(graph1_degree_deltas) / max(
        graph1_degree_deltas)
    graph2_max_degree = max([v for k, v in graph2.degree])
    graph2_degree_deltas = [max_degree - v for k, v in graph2.degree]
    graph2_degree_centralization = sum(graph2_degree_deltas) / max(
        graph2_degree_deltas)
    result['degree_centralization'] = abs(
        graph1_degree_centralization - graph2_degree_centralization)
    result['density'] = abs(nx.density(graph1) - nx.density(graph2))

    return result

if __name__ == '__main__':
    # Example usage of compare function
    n1, p1 = 100, 0.8
    n2, p2 = 100, 0.7

    graph1 = nx.gnp_random_graph(n1, p1, seed=93)
    graph2 = nx.gnp_random_graph(n2, p2, seed=95)

    print(compare(graph1, graph2))
