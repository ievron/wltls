"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import networkx as nx

from aux_lib import print_debug
from graphs import Graph
from math import log

###################################################################################################
# A trellis graph with (requiredPathsNum) paths and a given slice width (noted as b in the paper).
###################################################################################################
class TrellisGraph(Graph):
    G = None
    source = -1
    sink = -1
    edgesNum = 0
    verticesNum = 0
    pathNum = 0
    edgesFromSlice = None
    _shortcutEdges = None

    def __init__(self, requiredPathsNum, sliceWidth=2):

        self.sliceWidth = sliceWidth
        # Adding epsilon to fix numeric side effects such as log(1000,10)
        self.slices = int(log(requiredPathsNum, self.sliceWidth) + 0.0001)
        self.pathNum = requiredPathsNum
        self.edgesFromSlice = [[] for _ in range(self.slices + 2)]
        self._shortcutEdges = {}

        super(TrellisGraph, self).__init__()

        print_debug("Created a trellis graph with a slice width of b={}.".format(self.sliceWidth))

    # Checks if the edge is a shortcut edge to the sink (see paper)
    def isEdgeShortcut(self, edge):
        return self._shortcutEdges[edge]

    def _getInnerVertexIdx(self, slice, level):
        return slice * self.sliceWidth + level + 1

    # Returns slice and level of the vertex
    def _getVertexLocationByIdx(self, idx):
        return {'slice': int((idx - 1) / self.sliceWidth),
                'level': int((idx - 1) % self.sliceWidth)}

    # Creates the trellis graph (thoroughly explained in the paper)
    def _createGraph(self):
        from aux_lib import int2base
        self.G = nx.DiGraph()
        self.source = 0

        edges = []

        # Create source to first slice
        for l in range(self.sliceWidth):
            edge = (self.source, self._getInnerVertexIdx(0, l))
            self._addEdge(edges, edge, 0)

        # Create edges between slices
        for i in range(0, self.slices - 1):
            for l in range(0, self.sliceWidth):
                for nextL in range(self.sliceWidth):
                    edge = (self._getInnerVertexIdx(i, l), self._getInnerVertexIdx(i + 1, nextL))
                    self._addEdge(edges, edge, i + 1)

        # Add extra edges to sink to overcome path numbers which aren't a power of 2
        baseRepresentedPathNum = int2base(self.pathNum, self.sliceWidth)

        # Set sinks
        intermediateSinks = [self._getInnerVertexIdx(self.slices, l) for l in range(0, int(baseRepresentedPathNum[0]))]
        self.sink = max(intermediateSinks) + 1

        # Create intermediate sinks, to handle the situation where the first bit is not one
        for intermediateSink in intermediateSinks:
            # Create last slice to intermediate sinks
            for l in range(self.sliceWidth):
                edge = (self._getInnerVertexIdx(self.slices - 1, l), intermediateSink)
                self._addEdge(edges, edge, -2)

        # Create intermediate sinks to sink
        for intermediateSink in intermediateSinks:
            edge = (intermediateSink, self.sink)
            self._addEdge(edges, edge, -1)

        # Create shortcuts to sink
        for i in range(0, self.slices):
            level = baseRepresentedPathNum[-(i+1)]
            for l in range(0, level):
                edge = (self._getInnerVertexIdx(i, l), self.sink)
                self._addEdge(edges, edge, i + 1, True)

        self.G.add_edges_from(edges)

        return self.G.number_of_nodes(), edges

    def _addEdge(self, edges, edge, sourceSlice, isShortcut=False):
        edges.append(edge)
        self.edgesFromSlice[sourceSlice].append(edge)

        self._shortcutEdges[edge] = isShortcut

    # Get vertices (plotted) positions (list of tuples)
    def _getVerticesPositions(self):
        positions = []

        for v in self.G.nodes():
            if v == self.source:
                positions.append((0, 0))
            elif v == self.sink:
                # Real sink
                positions.append((self.slices + 3, 1.5))
            else:
                loc = self._getVertexLocationByIdx(v)

                # Normalize positions from -1 to 1
                positions.append((loc['slice'] + 1,
                                  1 - 2 * loc['level'] / (self.sliceWidth - 1)))

        return positions
