"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import abc

#############################################################################################
# An abstract graph.
#############################################################################################
class Graph(metaclass=abc.ABCMeta):
    edgesNum = None
    verticesNum = None
    _edges = None

    def __init__(self):
        verticesNum, edges = self._createGraph()
        self._edges = edges
        self.edgesNum = len(edges)
        self.verticesNum = verticesNum

    def edges(self):
        return self._edges

    # Get vertices (plotted) positions (list of tuples)
    @abc.abstractmethod
    def _getVerticesPositions(self):
        raise NotImplementedError

    # Creates the graph (should return the number of vertices and the set of edges
    @abc.abstractmethod
    def _createGraph(self):
        raise NotImplementedError

    # Plot the graph
    def show(self, W=None, block:bool=False, shouldLabelNodes=True, precisionDigits=1):
        import networkx as nx
        import matplotlib.pyplot as plt
        plt.clf()
        plt.tight_layout()

        nx.draw_networkx_nodes(self.G, pos=self._getVerticesPositions(), node_color='w')

        # Add weight labels to edges
        if W is not None:
            edgeLabels = { e: '{:.{}f}'.format(W[e], precisionDigits) for e in self.G.edges() }
        else:
            edgeLabels = {}


        nx.draw_networkx_edge_labels(self.G, pos=self._getVerticesPositions(), edge_labels=edgeLabels,
                                     font_size=10, label_pos=0.6)


        if shouldLabelNodes:
            nx.draw_networkx_labels(self.G, pos=self._getVerticesPositions(),
                                    font_size=6, font_weight="bold")

        self.G.graph['graph'] = {'rankdir': 'TD'}
        self.G.graph['node'] = {'shape': 'circle'}
        self.G.graph['edges'] = {'color': 'b', 'arrowsize': '4.0'}

        nx.draw_networkx_edges(self.G, pos=self._getVerticesPositions(), arrowsize=1.0)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')

        if block:
            fig = plt.gcf()
            fig.canvas.set_window_title("Close to continue")

        # Show
        plt.show(block=block)

        if not block:
            plt.waitforbuttonpress()

        return
