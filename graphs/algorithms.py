"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import numpy as np
import networkx as nx
import heapq
from itertools import islice

# Gets a networkx DAG, and finds its K heaviest paths in O(K(V+E)) - Viterbi
def DAG_k_heaviest_path_lengths(G, W, s, k=1, topologicalSort = None):
    # If only one path is needed, call a more efficient method
    if k == 1:
        return DAG_heaviest_path_lengths(G, W, s, topologicalSort)

    # Compute a topological sort only once
    if topologicalSort is None:
        topologicalSort = list(nx.topological_sort(G))

    lengths = {v: [(-np.infty, None)] for v in topologicalSort}

    # We sort at every iteration instead of using sorted list because apparently it's faster
    # Every list should be saved sorted
    lengths[s] = [(0, None)]

    # Viterbi algorithm, start from after the source
    for v in topologicalSort[topologicalSort.index(s) + 1:]:
        predecessors = G.predecessors(v)

        if not predecessors:
            continue

        lengths[v] = list(islice(heapq.merge(*[[(length[0] + W[(p, v)], (p, i)) for i,length in enumerate(lengths[p])]
                                            for p in predecessors],
                                            key=lambda x: x[0], reverse=True), k))


    return {key: [v[1] for v in val] for key, val in lengths.items()}

# A faster version for k = 1
# Gets a networkx DAG, and finds its heaviest path in O(V+E) - Viterbi
def DAG_heaviest_path_lengths(G, W, s, topologicalSort = None):
    if topologicalSort is None:
        topologicalSort = list(nx.topological_sort(G))

    lengths = {v: (-np.infty, None) for v in topologicalSort}

    # We sort at every iteration instead of using sorted list because apparently it's faster
    # Every list should be saved sorted
    lengths[s] = (0, None)

    # Viterbi algorithm, start from after the source
    for v in topologicalSort[topologicalSort.index(s) + 1:]:
        predecessors = G.predecessors(v)

        if not predecessors:
            continue

        maxVal = (-np.infty, None)
        for p in predecessors:
            currLength = lengths[p][0] + W[(p, v)]

            if currLength > maxVal[0]:
                maxVal = (currLength, (p, 0))

        lengths[v] = maxVal

    return {vertex: [parent[1]] for vertex, parent in lengths.items()}

# Restore the path from the dictionary of parents
def restorePathsFromParents(parents, pathsNum:int, s, t, W=None):
    # Note: We use s and t as if they were indices,
    # but it's only because the graph's nodes have incremental numbers as their ids
    if pathsNum > 1:
        assert W is not None, "To restore multiple paths, the weights matrix is mandatory"

    # Restore as many paths as required
    for currPath in range(0, pathsNum):
        currVertex = t
        requiredPathIdx = currPath

        path = []

        # Continue "recursively" until reaching the source s
        while currVertex != s:
            currParent, requiredPathIdx = parents[currVertex][requiredPathIdx]

            # We use append and reversed because it's supposedly much faster than insert(0, ...)
            path.append((currParent, currVertex))

            currVertex = currParent

        yield reversed(path)


