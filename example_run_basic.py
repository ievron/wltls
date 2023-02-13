from WLTLS.datasets import read
from WLTLS.decoding import HeaviestPaths, expLoss, squaredLoss, squaredHingeLoss
from WLTLS.mainModels.finalModel import FinalModel
from graphs import TrellisGraph
import argparse
import warnings
import numpy as np
from WLTLS.mainModels import WltlsModel
from WLTLS.learners import AveragedPerceptron, AROW
from aux_lib import Timing, print_debug
from WLTLS.codeManager import GreedyCodeManager, RandomCodeManager, FixedCodeManager
from WLTLS.datasets import datasets
from WLTLS.precomputedAssignments import precomputedAssignments
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
ASSIGNMENT_FIXED = "fixed"

def printSeparator():
    print("=" * 100)

all_datasets = [d.name for d in datasets.getAll()]

# Paths
DATA_PATH = "C:/Data"
MODEL_PATH = "C:/Models/"

# Parameters for run
DATASET_NAME = ["sector", "aloi_bin", "LSHTC1", "Dmoz", "imageNet"][1]
BINARY_LEARNER = [LEARNER_AROW, LEARNER_PERCEPTRON][0]
RANDOM_SEED = 2023
GRAPH_WIDTH = 2
ASSIGNMENT = [ASSIGNMENT_RANDOM, ASSIGNMENT_GREEDY, ASSIGNMENT_FIXED][2]
DECODING_LOSS = [LOSS_EXP, LOSS_SQUARED_HINGE, LOSS_SQUARED][0]
EXPERIMENT_SPARSE = False

DATASET = datasets.getParams(DATASET_NAME)
EPOCHS = DATASET.epochs
LOG_PATH = os.path.join(MODEL_PATH, "model")

warnings.filterwarnings("ignore",".*GUI is implemented.*")

printSeparator()
print("Learning a Wide-LTLS model, proposed in:")
print("Efficient Loss-Based Decoding On Graphs For Extreme Classification. NIPS 2018.")
printSeparator()

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load dataset
Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, LABELS, DIMS = read(DATA_PATH, DATASET)

printSeparator()


# Determine the loss
if DECODING_LOSS == LOSS_EXP:
    loss = expLoss
elif DECODING_LOSS == LOSS_SQUARED_HINGE:
    loss = squaredHingeLoss
else:
    loss = squaredLoss


if BINARY_LEARNER == LEARNER_AROW:
    learner = AROW
else:
    learner = AveragedPerceptron

print_debug("Using {} as the binary classifier.".format(BINARY_LEARNER))
print_debug("Decoding according to the {} loss.".format(DECODING_LOSS))



# Create the graph
trellisGraph = TrellisGraph(LABELS, GRAPH_WIDTH)
heaviestPaths = HeaviestPaths(trellisGraph, loss=loss)

# Process arguments
if ASSIGNMENT == ASSIGNMENT_FIXED and (DATASET_NAME in precomputedAssignments):
    codeManager = FixedCodeManager(LABELS, heaviestPaths.allCodes(), precomputedAssignments[DATASET_NAME])
elif ASSIGNMENT == ASSIGNMENT_GREEDY:
    # This is the greedy policy from the 2018 paper (not the 2023 one!)
    codeManager = GreedyCodeManager(LABELS, heaviestPaths.allCodes())
else:
    ASSIGNMENT = ASSIGNMENT_RANDOM
    codeManager = RandomCodeManager(LABELS, heaviestPaths.allCodes())

# Create the model
mainModel = WltlsModel(LABELS, DIMS, learner, codeManager, heaviestPaths, step_size = DATASET.step_size)

printSeparator()

LOG_PATH = os.path.join(MODEL_PATH, "{}_b{}_{}_epchs{}".format(
    DATASET_NAME.lower(), GRAPH_WIDTH, ASSIGNMENT, EPOCHS))

# Run the experiment
Experiment(mainModel, EPOCHS).run(Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest,
                                  computeAverageLoss=False,
                                  modelLogPath=LOG_PATH,
                                  earlyStop=DATASET.early_stop,
                                  forceStop=True,
                                  returnBestValidatedModel=DATASET.return_best)


printSeparator()

# Create a final model (for fast inference) and test it
finalModel = FinalModel(DIMS, mainModel, codeManager, heaviestPaths)
del mainModel

result = finalModel.test(Xtest, Ytest)

print_debug("The final model was tested in {} and achieved {:.1f}% accuracy.".format(
    Timing.secondsToString(result["time"]), result["accuracy"]
))

printSeparator()

if EXPERIMENT_SPARSE:
    print_debug("Experimenting sparse models:")

    from WLTLS.sparseExperiment import SparseExperiment

    ex = SparseExperiment(codeManager, heaviestPaths)
    ex.run(finalModel.W, Xtest, Ytest, Xvalid, Yvalid)

    printSeparator()
