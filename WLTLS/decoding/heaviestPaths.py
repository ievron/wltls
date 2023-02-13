"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

from WLTLS.decoding.decodingLosses import expLoss
from graphs import Graph, DAG_k_heaviest_path_lengths, restorePathsFromParents
import networkx as nx
import numpy as np
from aux_lib import pairwise, print_debug
from scipy.sparse import csr_matrix

#########################################################################
# Deccodes ECOC using a directed a-cyclic graph
#########################################################################
class HeaviestPaths():
    _DAG = None
    _edgeIndices = None
    _topologicalSort = None
    loss = None

    def __init__(self, DAG : Graph, loss = expLoss):
        self._DAG = DAG
        self._edges = list(self._DAG.edges())
        self.loss = loss

        # Assure the graph is dag, we don't check to save time
        assert (isinstance(self._DAG.G, nx.DiGraph) and nx.is_directed_acyclic_graph(self._DAG.G)), \
            "Graph must be directed a-cyclic"

        # Pre-compute the topological sort so we don't need to re-calculate it at every step
        self._topologicalSort = list(nx.topological_sort(self._DAG.G))

        self._edgeIndices = { e: idx for idx, e in enumerate(self.edges()) }

        self._edgesNum = DAG.edgesNum
        self._verticesNum = DAG.verticesNum

        self.sandboxW = np.zeros((self._verticesNum, self._verticesNum))

        self.initLossMatrix()

        print_debug("Created a heaviest path decoder with {} edges.".format(self._edgesNum))

    # Create an auxiliary binary matrix for *fast* calculation of losses (preparing weights)
    # Useful to employ the reduction to *any* loss function (See Section 5 in the paper)
    def initLossMatrix(self):
        self.lossMat = np.zeros((self._edgesNum,self._edgesNum))

        for i,slice in enumerate(self._DAG.edgesFromSlice):
            for e in slice:
                if not self._DAG.isEdgeShortcut(e):
                    for e2 in slice:
                        if e != e2:
                            self.lossMat[self._edgeIndices[e], self._edgeIndices[e2]] = 1
                else:
                    # If the edge is a shortcut,
                    # it should include (negative) terms from all edges which are farther from the source.
                    # See Appendix "Loss-based decoding generalization" for more details.
                    for slice2 in self._DAG.edgesFromSlice[i:]:
                        for e2 in slice2:
                            # We shouldn't include a (negative) term for the edge itself
                            if e == e2:
                                continue

                            self.lossMat[self._edgeIndices[e], self._edgeIndices[e2]] = 1

        self.lossMat = csr_matrix(self.lossMat)

    # Efficiently calculate the loss-based weights (details in the paper) using matrix multiplication
    def prepareWeights(self, responses):
        posLoss = self.loss(responses, 1)
        negLoss = self.loss(responses, -1)

        cumulative = self.lossMat.dot(negLoss)

        res = -(posLoss + cumulative)

        # Fill the results in the matrix structure
        for e in self.edges():
            self.sandboxW[e] = res[self._edgeIndices[e]]

        return self.sandboxW

    # Finds the best k best codes of the graph given the responses
    def findKBestCodes(self, responses, k:int=1):
        W = self.prepareWeights(responses)

        parents = DAG_k_heaviest_path_lengths(self._DAG.G, W, self._DAG.source,
                                              k=k, topologicalSort=self._topologicalSort)

        paths = restorePathsFromParents(parents, k, self._DAG.source, self._DAG.sink, W)

        return [self._pathToCode(path) for path in paths]

    def edges(self):
        return self._edges

    # Returns all possible paths of the saved graph
    def allCodes(self):
        for path in nx.all_simple_paths(self._DAG.G, source=self._DAG.source, target=self._DAG.sink):
            yield self._pathToCode(pairwise(path))

    def _pathToCode(self, path):
        # Creates a -1,1 matrix of the paths over the edges
        code = [-1] * self._edgesNum
        for e in path:
            code[self._edgeIndices[e]] = 1

        return code