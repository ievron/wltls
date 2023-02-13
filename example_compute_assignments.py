from WLTLS.codeManager import FixedCodeManager
from WLTLS.datasets import read
import time
from WLTLS.decoding import HeaviestPaths
from aux_lib import print_debug
from graphs import TrellisGraph
import numpy as np
import networkx as nx
from scipy import sparse
from WLTLS.datasets import datasets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform, pdist


LINE_WIDTH = 80
def printSeparator():
    print("=" * LINE_WIDTH)

def computeClassMeans(X, Y):
    print_debug("Computing class means (using input features).")
    means = np.zeros((LABELS, X.shape[1]), dtype=np.float32)

    for label in range(LABELS):
        dataFromLabel = X[Y == label, :]
        labelMean = dataFromLabel.mean(axis=0)
        means[label, :] = labelMean

    print_debug("Created {} means, each of {} features.".format(means.shape[0], means.shape[1]))

    return means

def createClassTaxonomy(distances):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,
                                    linkage="average", affinity="precomputed",
                                    compute_distances=True)
    model = model.fit(distances)

    G = nx.DiGraph()
    n_samples = means.shape[0]
    for i in range(n_samples):
        G.add_node(i)

    for i, inner_children in enumerate(model.children_):
        for inner_child in inner_children:
            G.add_edge(i + n_samples, inner_child)

    return G, max(G.nodes)


DATASET_PATH = r"C:/Data"
SLICE_WIDTH = 2
DATASET = datasets.getParams("aloi_bin")


# ===== Load dataset
Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, LABELS, DIMS = read(DATASET_PATH, DATASET)

printSeparator()

# ===== Compute class means
means = computeClassMeans(Xtrain, Ytrain)

# ===== Compute distance matrix (between classes) using class means
print_debug("Computing {0}x{0} distance matrix.".format(means.shape[0]))
means = sparse.csr_matrix(means)
Dcls = pairwise_distances(X=means, Y=means, metric='euclidean')

# ===== Compute a class taxonomy in the form of a networkx tree.
# This part should be replaced if you already have an available hierarchy.
# Can become cumbersome for many classes (due to the high dimensionality and many classes in XMC)
G, root = createClassTaxonomy(Dcls)

# ===== Compute a greedy similarity-preserving codeword-to-class assignment.
# (see Section 4.2 and Appendix F in Evron et al. 2023.)
similarityAssignment = []
for v in nx.dfs_preorder_nodes(G, source=root):
    if int(v) < means.shape[0]:
        similarityAssignment.append(int(v))


# Print assignment (copy this to other precomputedAssignments.py if you want to run a full simulation with it.)
print_debug("The found assignment is:")
print(similarityAssignment)



# ===== Create W-LTLS modules
trellisGraph = TrellisGraph(LABELS, SLICE_WIDTH)
heaviestPaths = HeaviestPaths(trellisGraph)

# Create a fixed code manager using the precomputed assignment
codeManager = FixedCodeManager(LABELS, heaviestPaths.allCodes(), similarityAssignment)


# ===== Showing some statistics (this entire part is not mandatory)

# Apply codebook assignment
codebook = codeManager.codingMatrix.astype(np.float32)[np.argsort(similarityAssignment), :]

print_debug("Computing the distances between codewords")
t1 = time.time()
Dcw = squareform(pdist(codebook, 'minkowski', p=1))
print_debug("Computation took {:.1f}sec, Matrix shape is {} and weight is {:,.1f}MB".format(
    time.time() - t1, Dcw.shape, Dcw.nbytes / 1024 ** 2))

assert Dcw.shape == Dcls.shape, "Distance matrix (between classes and between codewords) should be of the same shape."

# Normalizing the distance matrices
Dcls = Dcls / np.linalg.norm(Dcls)
Dcw = Dcw / np.linalg.norm(Dcw)

# Comparing the codeword-to-class scores (Eq. 5 in Evron et al. 2023) of the similarity-preserving assignment
# and random assignments (random should have higher scores)


print_debug("Smart assignment: {:.2e}".format(np.linalg.norm(Dcls - Dcw) ** 2))

randoms = []
for _ in range(5):
    permutation = np.random.permutation(Dcls.shape[0])
    Dcls = Dcls[permutation, :]
    Dcls = Dcls[:, permutation]

    dist = np.linalg.norm(Dcls - Dcw) ** 2
    randoms.append(dist)

print_debug("Mean of 5 random assignments: {:.2e}, Std: {:.2e}".format(np.mean(randoms), np.std(randoms)))





