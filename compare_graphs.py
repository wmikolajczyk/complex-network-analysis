# -*- coding: utf-8
import networkx as nx

from collections import OrderedDict

from scipy import stats


MEASUREMENTS = {
    'list': [
        'degree_centrality',
        'closeness_centrality',
        'betweenness_centrality',
        'pagerank',
    ],
    'value': [
        'average_shortest_path_length',
        'diameter',
        'degree_centralization',
        'density',
    ]
}


def get_graph_measurements(graph):
    """
    Returns graph measurements dict
    """
    graph_measurements = OrderedDict()

    # measurement is a list of node values
    graph_measurements['degree_centrality'] = list(nx.degree_centrality(graph).values())
    graph_measurements['closeness_centrality'] = list(nx.closeness_centrality(graph).values())
    graph_measurements['betweenness_centrality'] = list(nx.betweenness_centrality(graph).values())
    graph_measurements['pagerank'] = list(nx.pagerank(graph).values())
    # measurement is a number
    try:
        graph_measurements['average_shortest_path_length'] = nx.average_shortest_path_length(graph)
    except nx.NetworkXError as e:
        graph_measurements['average_shortest_path_length'] = None
        print('Cannot compute average_shortest_path_length - {}'.format(e))
    try:
        graph_measurements['diameter'] = nx.diameter(graph)
    except nx.NetworkXError as e:
        graph_measurements['diameter'] = None
        print('Cannot compute diameter - {}'.format(e))

    max_degree = max([v for k, v in graph.degree])
    degree_deltas = [max_degree - v for k, v in graph.degree]
    graph_measurements['degree_centralization'] = sum(degree_deltas) / max(degree_deltas)
    graph_measurements['density'] = nx.density(graph)

    return graph_measurements


def compare_measurements(graph1_measurements, graph2_measurements):
    """
    Calculates statistical KS and p-values for list of node values
    and absolute measurements subtraction result for numeric measurements
    """
    results = OrderedDict()
    for measurement in MEASUREMENTS['list']:
        results[measurement] = stats.ks_2samp(
            graph1_measurements[measurement], graph2_measurements[measurement])

    for measurement in MEASUREMENTS['value']:
        try:
            results[measurement] = abs(
                graph1_measurements[measurement] - graph2_measurements[measurement])
        except TypeError:
            # results[measurement] = None
            pass
    return results


def average_comparison(comparison_list):
    comparison_result = {
        measurement: {'num_of_items': 0, 'value': 0}
        for measurement in MEASUREMENTS['list'] + MEASUREMENTS['value']
    }

    for comparison in comparison_list:
        for key, value in comparison.items():
            # Ks_2sampResult -> KS statistic and p-value - choose p-value
            if isinstance(value, stats.stats.Ks_2sampResult):
                value = value.pvalue
            comparison_result[key]['num_of_items'] += 1
            comparison_result[key]['value'] += value

    result = {}
    for key in comparison_result:
        try:
            result[key] = comparison_result[key]['value'] / comparison_result[key]['num_of_items']
        except ZeroDivisionError:
            result[key] = None
    return result
