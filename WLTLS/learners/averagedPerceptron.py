"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

import numpy as np
from WLTLS.learners.learner import Learner

#############################################################################################
# An implementation of the averaged Perceptron.
# Similar algorithm in [Log-time and Log-space Extreme Classification; Jasinska et al. 2016].
#############################################################################################
class AveragedPerceptron(Learner):
    isFinal = False
    margin = 1

    updatesCounter = 1

    def __init__(self, DIM):
        super().__init__(DIM)

        self.W = np.zeros((self.DIM,), dtype=np.float32)
        self.WA = np.zeros((self.DIM,), dtype=np.float32)

    def getModelSize(self):
        return self.W.nbytes
        # Similarly to the convention of other papers, we consider only the size of the actual separators,
        # because they are the ones which will be used during test time.
        # The alternative is:
        # return self.W.nbytes + self.WA.nbytes

    def refit(self, x, y, actualMargin):
        self.updatesCounter += 1

        # Multi-class:
        # Check if the top negative code is not far enough from the actual code
        if (y * actualMargin) <= self.margin:
            # It's important to use the sparsity of x
            res = np.multiply(x.data, y)

            self.W[x.indices] += res
            self.WA[x.indices] += self.updatesCounter * res

    def score(self, x):
        # Remember to use the sparsity, and also don't transpose
        return np.dot(x.data, self.W[x.indices])

    def eval(self):
        super().eval()

    # "Unaverage" the weights
    def _prepareToTrain(self):
        self.WA /= self.updatesCounter
        self.W += self.WA
        self.WA *= self.updatesCounter

    # Average the weights
    def _prepareToEval(self):
        self.WA /= self.updatesCounter
        self.W -= self.WA
        self.WA *= self.updatesCounter
