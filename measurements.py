import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict

from collections import OrderedDict
from scipy import stats


MEASUREMENTS = OrderedDict([
    ('degree_centrality', 'list'),  # in_degree_centrality / out_degree_centrality?
    ('closeness_centrality', 'list'),
    ('betweenness_centrality', 'list'),
    ('pagerank', 'list'),

    ('average_shortest_path_length', 'value'),
    ('diameter', 'value'),

    ('degree_centralization', 'value'),
    ('closeness_centralization', 'value'),
    ('betweenness_centralization', 'value'),
    ('pagerank_centralization', 'value'),
    # not implemented for directed graphs
    # ('clustering_centralization', 'value'),

    ('density', 'value'),
    ('degree_assortativity', 'value'),
    ('reciprocity', 'value'),
    ('transitivity', 'value'),
])

# weight=None - to explicitly show to treat graph like there is no weights on edges
#   because there are not - in the recreated graph


def get_graph_measurements(graph):
    """
    Returns graph measurements dict
    """
    def freeman_centralization(values):
        max_val = max(values)
        deltas = [max_val - val for val in values]

        return sum(deltas) / max(deltas)

    graph_measurements = OrderedDict()

    # measurement is a list of node values
    graph_measurements['degree_centrality'] = list(nx.degree_centrality(graph).values())
    graph_measurements['closeness_centrality'] = list(nx.closeness_centrality(graph).values())
    graph_measurements['betweenness_centrality'] = list(nx.betweenness_centrality(graph, weight=None).values())
    graph_measurements['pagerank'] = list(nx.pagerank(graph, weight=None).values())
    # measurement is a number
    try:
        graph_measurements['average_shortest_path_length'] = nx.average_shortest_path_length(graph, weight=None)
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
    except ZeroDivisionError as e:
        graph_measurements['degree_centralization'] = None
        print('Cannot compute degree centralization - {}'.format(e))
    graph_measurements['closeness_centralization'] = freeman_centralization(graph_measurements['closeness_centrality'])
    graph_measurements['betweenness_centralization'] = freeman_centralization(graph_measurements['betweenness_centrality'])
    graph_measurements['pagerank_centralization'] = freeman_centralization(graph_measurements['pagerank'])
    # not implemented for directed graphs
    # graph_measurements['clustering_centralization'] = freeman_centralization(nx.clustering(graph).values())

    graph_measurements['density'] = nx.density(graph)
    graph_measurements['degree_assortativity'] = nx.degree_assortativity_coefficient(graph, weight=None)
    graph_measurements['reciprocity'] = nx.reciprocity(graph)
    graph_measurements['transitivity'] = nx.transitivity(graph)

    return graph_measurements


def collect_graph_measurements(graph1_measurements, graph2_measurements):
    results = OrderedDict()
    for measurement, m_type in MEASUREMENTS.items():
        if m_type == 'list':
            ks_stat, ks_pval = stats.ks_2samp(
                graph1_measurements[measurement],
                graph2_measurements[measurement])
            ent_pred = stats.entropy(graph2_measurements[measurement])
            kl_div = stats.entropy(
                graph1_measurements[measurement], 
                graph2_measurements[measurement])

            results[measurement + '_ks_stat'] = ks_stat
            results[measurement + '_ks_pval'] = ks_pval 
            results[measurement + '_ent_pred'] = ent_pred
            results[measurement + '_kl_div'] = kl_div
        else:
            results[measurement] = graph2_measurements[measurement]
    return results


def get_measurements_results_df(df, orig_df):
    alpha = 0.95
    results = []
    for measure in df.columns:
        # drop NaNs
        values = df[measure].dropna()

        # compute the mean and the standard deviation 
        # of the measure from all experiments
        mean, sigma = np.mean(values), np.std(values)

        # compute confidence intervals of the mean 
        # of the measure from all experiments
        _lower, _upper = stats.norm.interval(alpha, 
                                       loc=mean, 
                                       scale=sigma/np.sqrt(len(values)))
        row = {
            'measure': measure, 
            'mean': mean, 
            'lower_endpoint': _lower, 
            'upper_endpoint': _upper, 
            'original': None,
            'is_between_bounds': None
        }
        if measure in orig_df.columns:
            row['original'] = orig_df[measure][0]
            if not np.isnan(row['original']):
                row['is_between_bounds'] = bool(_lower <= orig_df[measure][0] <= _upper)
        results.append(row)
    results_df = pd.DataFrame(results)
    results_df = results_df.reindex(columns=[
        'measure', 'mean', 'lower_endpoint', 'upper_endpoint', 'original', 
        'is_between_bounds'])
    return results_df


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


def print_graph_measurements(graph_measurements):
    msg = 'Graph measurements:\n'
    for measurement, m_type in MEASUREMENTS.items():
        # skip list of values
        if m_type == 'list':
            continue
        msg += '  {}: {}\n'.format(measurement, graph_measurements[measurement])
    print(msg)


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


def average_comparison(comparison_list):
    keys = comparison_list[0].keys()
    avg_dict = defaultdict(list)

    for comparison in comparison_list:
        for key in keys:
            avg_dict[key].append(comparison[key])

    for key in keys:
        avg_dict[key] = list(filter(None, avg_dict[key]))
        try:
            avg_dict[key] = sum(avg_dict[key]) / len(avg_dict[key])
        except ZeroDivisionError:
            avg_dict[key] = None

    return avg_dict
