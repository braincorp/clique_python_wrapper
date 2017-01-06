
import numpy as np

from clique_python_wrapper.clique import find_max_weighted_clique, find_max_weighted_independent_set


def test_find_max_weighted_clique():

    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[1]], dtype=np.uint8
            ),
            vertex_weights=np.array([1], dtype=np.int32)
        ),
        [True]
    )

    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[1]], dtype=np.uint8
            ),
            vertex_weights=np.array([2], dtype=np.int32)
        ),
        [True]
    )

    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 1],
                 [1, 0]
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 1], dtype=np.int32)
        ),
        [True, True]
    )

    # only upper part of the adjacency matters
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[1, 1],
                 [0, 0]
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 1], dtype=np.int32)
        ),
        [True, True]
    )

    # pick the first vertex as a clique since weights are the same
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 0],
                 [0, 0]
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 1], dtype=np.int32)
        ),
        [True, False]
    )

    # picke the second since weight is larger
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 0],
                 [0, 0]
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 2], dtype=np.int32)
        ),
        [False, True]
    )


    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 2, 3], dtype=np.int32)
        ),
        [False, False, True]
    )

    # fully connected 3x3 - all are in the clique
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0],
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 2, 3], dtype=np.int32)
        ),
        [True, True, True]
    )

    # fully connected 3x3 - all are in the clique
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 1, 1],
                 [0, 0, 1],
                 [0, 0, 0],
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 2, 3], dtype=np.int32)
        ),
        [True, True, True]
    )

    # the first and the second vertices are in the clique and it has weight 5
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([2, 3, 4], dtype=np.int32)
        ),
        [True, True, False]
    )

    # the first and the second vertices are in the clique but it has weight 5, so the last vertex with 6 is selected
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([2, 3, 6], dtype=np.int32)
        ),
        [False, False, True]
    )

    # fully connected 4x4 - all are in the clique
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 1, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0],
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([2, 3, 4, 5], dtype=np.int32)
        ),
        [True, True, True, True]
    )

    # Two cliques (0+2 with weight 6 and 1+3 with weight 8)
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([2, 3, 4, 5], dtype=np.int32)
        ),
        [False, True, False, True]
    )

    # Two cliques (0+2 with weight 9 and 1+3 with weight 8)
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([2, 3, 7, 5], dtype=np.int32)
        ),
        [True, False, True, False]
    )

    # 3 cliques
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0],
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([2, 3, 4, 5], dtype=np.int32)
        ),
        [False, False, True, True]
    )

    # two separated cliques in one graph (second has larger weight)
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 1, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0],
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 1, 1, 2, 1, 1], dtype=np.int32)
        ),
        [False, False, False, True, True, True]
    )

    # two separated cliques in one graph (second has larger weight) that are connected by one edge doesn't change the answer
    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 1, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0],
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 1, 1, 2, 1, 1], dtype=np.int32)
        ),
        [False, False, False, True, True, True]
    )

    np.testing.assert_array_equal(
        find_max_weighted_clique(
            adjacency_matrix=np.array(
                [[0, 0, 1],
                 [0, 0, 1],
                 [0, 0, 0],
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([2, 2, 1], dtype=np.int32)
        ),
        [False, True, True]
    )


def test_find_max_weighted_independent_set():
    np.testing.assert_array_equal(
        find_max_weighted_independent_set(
            adjacency_matrix=np.array(
                [[0]
                ], dtype=np.uint8
            ),
            vertex_weights=np.array([1], dtype=np.int32)
        ),
        [True]
    )

    # pick the first vertex as a clique since weights are the same
    np.testing.assert_array_equal(
        find_max_weighted_independent_set(
            adjacency_matrix=np.array(
                [[0, 0],
                 [0, 0]
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 1], dtype=np.int32)
        ),
        [True, True]
    )

    # pick the second vertex as the max one
    np.testing.assert_array_equal(
        find_max_weighted_independent_set(
            adjacency_matrix=np.array(
                [[0, 1],
                 [0, 0]
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 2], dtype=np.int32)
        ),
        [False, True]
    )

    # two separated cliques in one graph - pick single vertices with the largest weight
    np.testing.assert_array_equal(
        find_max_weighted_independent_set(
            adjacency_matrix=np.array(
                [[0, 1, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0],
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([1, 2, 1, 2, 1, 1], dtype=np.int32)
        ),
        [False, True, False, True, False, False]
    )

    # two separated cliques in one graph connected with and edge. The vertex with an edge has bigger weight,
    # but it has more edges
    np.testing.assert_array_equal(
        find_max_weighted_independent_set(
            adjacency_matrix=np.array(
                [[0, 1, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0],
                 ], dtype=np.uint8
            ),
            vertex_weights=np.array([3, 1, 4, 2, 1, 1], dtype=np.int32)
        ),
        [True, False, False, True, False, False]
    )


if __name__ == '__main__':
    test_find_max_weighted_clique()
    test_find_max_weighted_independent_set()
