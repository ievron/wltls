from WLTLS.datasets import read
from WLTLS.decoding import HeaviestPaths, expLoss, squaredLoss, squaredHingeLoss
from WLTLS.mainModels.finalModel import FinalModel
from graphs import TrellisGraph
import argparse
import warnings
import numpy as np
from WLTLS.mainModels import WltlsModel
from WLTLS.learners import AveragedPerceptron, AROW
from aux_lib import Timing
from WLTLS.codeManager import GreedyCodeManager, RandomCodeManager
from WLTLS.datasets import datasets
from WLTLS.experiment import Experiment
import os

# Const choices
LEARNER_AROW = "AROW"
LEARNER_PERCEPTRON = "perceptron"
LOSS_EXP = "exponential"
LOSS_SQUARED = "squared"
LOSS_SQUARED_HINGE = "squared_hinge"
ASSIGNMENT_GREEDY = "greedy"
ASSIGNMENT_RANDOM = "random"

LINE_WIDTH = 80
def printSeparator():
    print("=" * LINE_WIDTH)

all_datasets = [d.name for d in datasets.getAll()]

# Set argument parser
parser = argparse.ArgumentParser(description="Runs a single W-LTLS experiment. " +
                                             "See https://github.com/ievron/wltls/ for documentation and license details.")
parser.add_argument("dataset",
                    choices=all_datasets,
                    help="Dataset name")
parser.add_argument("data_path", help="Path of the directory holding the datasets downloaded from PD-Sparse")
parser.add_argument("model_dir", help="Path of a directory to save the model in (model.npz)")
parser.add_argument("-slice_width", type=int, help="The slice width of the trellis graph", default=2)
parser.add_argument("-decoding_loss",
                    choices=[LOSS_EXP, LOSS_SQUARED_HINGE, LOSS_SQUARED],
                    nargs="?",
                    const=LOSS_EXP,
                    default=LOSS_EXP,
                    help="The loss for the loss-based decoding scheme")
parser.add_argument("-epochs", type=int, help="Number of epochs", default=-1)
parser.add_argument("-rnd_seed", type=int, help="Random seed")
parser.add_argument("-path_assignment", choices=[ASSIGNMENT_RANDOM, ASSIGNMENT_GREEDY],
                    nargs="?", const=ASSIGNMENT_RANDOM, help="Path assignment policy", default=ASSIGNMENT_RANDOM)
parser.add_argument("-binary_classifier", choices=[LEARNER_AROW, LEARNER_PERCEPTRON],
                    nargs="?", const=LEARNER_AROW,
                    help="The binary classifier for learning the binary subproblems",
                    default=LEARNER_AROW)
parser.add_argument("--plot_graph", dest='show_graph', action='store_true', help="Plot the trellis graph on start")
parser.add_argument("--sparse", dest='try_sparse', action='store_true',
                    help="Experiment sparse models at the end of training")
parser.set_defaults(show_graph=False)

args = parser.parse_args()

# If user gave a random seed
if args.rnd_seed is not None:
    import random
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)

from WLTLS.datasets import datasets
DATASET =   datasets.getParams(args.dataset)
EPOCHS =    args.epochs if args.epochs >= 1 else DATASET.epochs
LOG_PATH = os.path.join(args.model_dir, "model")

warnings.filterwarnings("ignore",".*GUI is implemented.*")

printSeparator()
print("Learning a Wide-LTLS model, proposed in:")
print("Efficient Loss-Based Decoding On Graphs For Extreme Classification. NIPS 2018.")
printSeparator()

# Load dataset
Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, LABELS, DIMS = read(args.data_path, DATASET)

printSeparator()

assert args.slice_width >= 2 and args.slice_width <= LABELS,\
    "Slice width must be larger than 1 and smaller than the number of classes."


# Decide the loss
if args.decoding_loss == LOSS_EXP:
    loss = expLoss
elif args.decoding_loss == LOSS_SQUARED_HINGE:
    loss = squaredHingeLoss
else:
    loss = squaredLoss

# Create the graph
trellisGraph = TrellisGraph(LABELS, args.slice_width)
heaviestPaths = HeaviestPaths(trellisGraph, loss=loss)

# Plot the graph if needed
if args.show_graph:
    trellisGraph.show(block=True)

# Process arguments
if args.path_assignment == ASSIGNMENT_RANDOM:
    codeManager = RandomCodeManager(LABELS, heaviestPaths.allCodes())
else:
    codeManager = GreedyCodeManager(LABELS, heaviestPaths.allCodes())

if args.binary_classifier == LEARNER_AROW:
    learner = AROW
else:
    learner = AveragedPerceptron

print("Using {} as the binary classifier.".format(args.binary_classifier))
print("Decoding according to the {} loss.".format(args.decoding_loss))

# Create the model
mainModel = WltlsModel(LABELS, DIMS, learner, codeManager, heaviestPaths)

printSeparator()

# Run the experiment
Experiment(mainModel, EPOCHS).run(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest,
                                  modelLogPath=LOG_PATH,
                                  returnBestValidatedModel=True)


printSeparator()

# Create a final model (for fast inference) and test it
finalModel = FinalModel(DIMS, mainModel, codeManager, heaviestPaths)
del mainModel

result = finalModel.test(Xtest, Ytest)

print("The final model was tested in {} and achieved {:.1f}% accuracy.".format(
    Timing.secondsToString(result["time"]), result["accuracy"]
))

printSeparator()

# Check if we want to experiment sparse models
if args.try_sparse:
    print("Experimenting sparse models:")

    from WLTLS.sparseExperiment import SparseExperiment

    ex = SparseExperiment(codeManager, heaviestPaths)
    ex.run(finalModel.W, Xtest, Ytest, Xvalid, Yvalid)

    printSeparator()
