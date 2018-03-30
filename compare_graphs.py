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
    result['degree_centrality_delta'] = stats.ks_2samp(graph1_degree_centrality,
                                                 graph2_degree_centrality)

    graph1_closeness_centrality = list(
        nx.closeness_centrality(graph1).values())
    graph2_closeness_centrality = list(
        nx.closeness_centrality(graph2).values())
    result['closeness_centrality_delta'] = stats.ks_2samp(
        graph1_closeness_centrality, graph2_closeness_centrality)

    graph1_betweenness_centrality = list(
        nx.betweenness_centrality(graph1).values())
    graph2_betweenness_centrality = list(
        nx.betweenness_centrality(graph2).values())
    result['betweenness_centrality_delta'] = stats.ks_2samp(
        graph1_betweenness_centrality, graph2_betweenness_centrality)

    graph1_pagerank = list(nx.pagerank(graph1).values())
    graph2_pagerank = list(nx.pagerank(graph2).values())
    result['pagerank_delta'] = stats.ks_2samp(graph1_pagerank, graph2_pagerank)

    # absolute value of global graph measurements subtraction
    try:
        result['average_shortest_path_length_delta'] = abs(
            nx.average_shortest_path_length(
                graph1) - nx.average_shortest_path_length(graph2))
    except nx.NetworkXError as e:
        # graph is not connected - it's possible for Watts-Strogats
        print('Cannot compute average_shortest_path_length - {}'.format(e))
    try:
        result['diameter_delta'] = abs(nx.diameter(graph1) - nx.diameter(graph2))
    except nx.NetworkXError as e:
        # graph is not connected - it's possible for Watts-Strogats
        print('Cannot compute diameter - {}'.format(e))
    graph1_max_degree = max([v for k, v in graph1.degree])
    graph1_degree_deltas = [graph1_max_degree - v for k, v in graph1.degree]
    graph1_degree_centralization = sum(graph1_degree_deltas) / max(
        graph1_degree_deltas)
    graph2_max_degree = max([v for k, v in graph2.degree])
    graph2_degree_deltas = [graph2_max_degree - v for k, v in graph2.degree]
    graph2_degree_centralization = sum(graph2_degree_deltas) / max(
        graph2_degree_deltas)
    result['degree_centralization_delta'] = abs(
        graph1_degree_centralization - graph2_degree_centralization)
    result['density_delta'] = abs(nx.density(graph1) - nx.density(graph2))

    return result
