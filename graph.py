import networkx as nx

from collections import OrderedDict
from scipy import stats


def generate_graph(graph_func, params):
    # set seed
    seed = 93
    # generate graph based on passed function and params
    graph = graph_func(**params, seed=seed)
    return graph


def attach_graph_attributes(graph):
    # get list of attributes for each node id
    degree_centralities = nx.degree_centrality(graph)
    betweenness_centralities = nx.betweenness_centrality(graph)
    closeness_centralities = nx.closeness_centrality(graph)
    pageranks = nx.pagerank(graph)

    # attach appropriate attributes to each node
    for node_id in graph.nodes:
        node_attributes = {
            'degree_centrality': degree_centralities[node_id],
            'betweenness_centrality': betweenness_centralities[node_id],
            'closeness_centrality': closeness_centralities[node_id],
            'pagerank': pageranks[node_id]
        }
        graph.node[node_id].update(node_attributes)


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

    try:
        graph_measurements['degree_centralization'] = freeman_centralization(graph_measurements['degree_centrality'])
        graph_measurements['closeness_centralization'] = freeman_centralization(graph_measurements['closeness_centrality'])
        graph_measurements['betweenness_centralization'] = freeman_centralization(graph_measurements['betweenness_centrality'])
        graph_measurements['pagerank_centralization'] = freeman_centralization(graph_measurements['pagerank'])
        #graph_measurements['clustering_centralization'] = freeman_centralization(nx.clustering(graph).values())
    except ZeroDivisionError as e:
        graph_measurements['degree_centralization'] = None
        print('Cannot compute degree centralization - {}'.format(e))

    graph_measurements['density'] = nx.density(graph)
    graph_measurements['degree_assortativity'] = nx.degree_assortativity_coefficient(graph)
    graph_measurements['reciprocity'] = nx.reciprocity(graph)
    graph_measurements['transitivity'] = nx.transitivity(graph)

    return graph_measurements


def freeman_centralization(values):
    max_val = max(values)
    deltas = [max_val - val for val in values]

    return sum(deltas) / max(deltas)


MEASUREMENTS = OrderedDict([
    ('degree_centrality', 'list'),
    ('closeness_centrality', 'list'),
    ('betweenness_centrality', 'list'),
    ('pagerank', 'list'),

    ('average_shortest_path_length', 'value'),
    ('diameter', 'value'),

    ('degree_centralization', 'value'),
    ('closeness_centralization', 'value'),
    ('betweenness_centralization', 'value'),
    ('pagerank_centralization', 'value'),
    # Not implemented for directed
    #('clustering_centralization', 'value'),

    ('density', 'value'),
    ('degree_assortativity', 'value'),
    ('reciprocity', 'value'),
    ('transitivity', 'value'),
])


def compare_graph_measurements(graph1_measurements, graph2_measurements):
    results = OrderedDict()
    for measurement, m_type in MEASUREMENTS.items():
        if m_type == 'list':
            val = stats.ks_2samp(
                graph1_measurements[measurement],
                graph2_measurements[measurement]).pvalue
        else:
            if (graph1_measurements[measurement] is None or
                graph2_measurements[measurement] is None) or graph1_measurements[measurement] == 0:
                val = None
            else:
                # abs in denominator is required because some measures can have values < 0
                val = abs(
                    graph1_measurements[measurement] -
                    graph2_measurements[measurement]) / abs(graph1_measurements[measurement])
        results[measurement] = val
    return results


def print_comparison_results(comparison_results):
    ks_test_min_threshold = 0.05
    abs_dist_max_threshold = 0.10

    for measurement, m_type in MEASUREMENTS.items():
        result = '{}: {}'.format(measurement, comparison_results[measurement])

        if m_type == 'list':
            name = '(KS test p-value)'
            passed_test = comparison_results[measurement] > ks_test_min_threshold
        else:
            name = '(abs distance)'
            if comparison_results[measurement] is not None:
                passed_test = comparison_results[measurement] < abs_dist_max_threshold
            else:
                passed_test = None
        msg = '{:18} {:50} [passed: {}]'.format(name, result, passed_test)
        print(msg)
