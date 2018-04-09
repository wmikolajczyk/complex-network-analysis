from generate_graph import generate_graph, recreate_graph
from compare_graphs import get_graph_measurements, compare_measurements


def analyse_graph_family(params_list, graph_func):
    result = []

    for params in params_list:
        original_graph = generate_graph(graph_func, params)
        recreated_graph = recreate_graph(original_graph)

        original_graph_measurements = get_graph_measurements(original_graph)
        recreated_graph_measurements = get_graph_measurements(recreated_graph)
        comparison = compare_measurements(original_graph_measurements, recreated_graph_measurements)

        result.append({
            'original': original_graph, 'recreated': recreated_graph,
            'params': params, 'comparison': comparison
        })

    return result
