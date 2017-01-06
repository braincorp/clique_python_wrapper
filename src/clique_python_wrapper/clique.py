
import numpy as np
from clique_python_wrapper._wrapper_max_weighted_clique_module import find_max_weighted_clique as find_max_weighted_clique_impl


def find_max_weighted_clique(adjacency_matrix, vertex_weights):
    '''
    Find max weighted clique of undirected graph with vertex weights.
    :param adjacency_matrix: np.uint8 square adjacency matrix of an undirected graph where non zero element mean an edge
        between vertices. Only upper triangular part of the matrix is used.
    :param vertex_weights: np.int32 vector of positive vertex weights
    :return: np.bool vector of graph size indicating whether vertex belong to the maximum weighted clique
    '''
    assert (adjacency_matrix.dtype == np.uint8)
    assert (vertex_weights.dtype == np.int32)
    assert (adjacency_matrix.shape[0] == adjacency_matrix.shape[1])
    assert (adjacency_matrix.shape[0] == vertex_weights.shape[0])
    if vertex_weights.min() <= 0:
        raise AssertionError("Clique algorithm works only with positive weights")

    # default algorithm parameters are hardcoded since other combinations are not tested
    max_clique_number = 1000
    bool_only_maximal = True
    min_weight = 0
    max_weight = 0

    best_hypothesis_int = find_max_weighted_clique_impl(
        adjacency_matrix, vertex_weights, min_weight, max_weight, bool_only_maximal, max_clique_number)

    return best_hypothesis_int > 0


def find_max_weighted_independent_set(adjacency_matrix, vertex_weights):
    '''
    Find max weighted independent set of undirected graph with vertex weights.
    :param adjacency_matrix: np.uint8 square adjacency matrix of an undirected graph where non zero element mean an edge
        between vertices. Only upper triangular part of the matrix is used.
    :param vertex_weights: np.int32 vector of positive vertex weights
    :return: np.bool vector of graph size indicating whether vertex belong to the maximum weighted independent set
    '''
    assert (adjacency_matrix.dtype == np.uint8)
    inverted_adjacency_matrix = np.zeros_like(adjacency_matrix)
    inverted_adjacency_matrix[adjacency_matrix == 0] = 1
    # since some algorithms use upper part of the adjacency matrix, we make it 0
    inverted_adjacency_matrix[np.tril_indices(adjacency_matrix.shape[0])] = 0
    return find_max_weighted_clique(inverted_adjacency_matrix, vertex_weights)
