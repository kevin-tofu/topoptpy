import numpy as np
from scipy.sparse import csr_matrix


def build_element_adjacency_matrix(mesh):
    """
    Returns sparse adjacency matrix A such that A[i, j] = 1
    if element i and j share at least one node.
    """
    num_elements = mesh.nelements
    rows, cols = [], []

    for i, elem_nodes in enumerate(mesh.t.T):
        for node in elem_nodes:
            connected_elements = np.where((mesh.t == node).any(axis=0))[0]
            for j in connected_elements:
                rows.append(i)
                cols.append(j)

    data = np.ones(len(rows), dtype=np.uint8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(num_elements, num_elements))

    return adjacency


def get_adjacent_elements(mesh, element_indices):
    """
    Given a list of element indices, return the set of elements
    that are adjacent (share at least one node) with any of them.
    """
    adjacency = build_element_adjacency_matrix(mesh)
    neighbors = set()

    for idx in element_indices:
        adjacent = adjacency[idx].nonzero()[1]
        neighbors.update(adjacent)

    # exlude original elements
    neighbors.difference_update(element_indices)

    return sorted(neighbors)
