# pylint: disable=missing-docstring
import networkx as nx


def find_max_k_truss(graph):
    # initialize the bounds of the binary search
    left = 2
    right = max(d for _, d in graph.degree())

    # perform the binary search
    while left <= right:
        mid = (left + right) // 2
        k_truss = nx.k_truss(graph, k=mid)
        if k_truss:
            # if the k-truss is not empty, try increasing k
            left = mid + 1
        else:
            # if the k-truss is empty, try decreasing k
            right = mid - 1

    # return the maximum k value for which the k-truss is not empty
    return right
