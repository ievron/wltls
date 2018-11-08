"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

import numpy as np
from scipy.stats import norm
from WLTLS.learners.learner import Learner

#############################################################################################
# Implementation of Adaptive regularization of weight vectors [Crammer et al. 2009]
#############################################################################################
class AROW(Learner):
    isFinal = False
    margin = 1 # Could be tuned
    _r = 1 # (see the paper of AROW)
    _eta = 0.85 # (see the paper of AROW)
    _phi = None

    def __init__(self, DIM):
        super().__init__(DIM)

        # Initialize the mean vector and covariance (diagonal) matrix
        self.mean = np.zeros((self.DIM,), dtype=np.float32)
        self.covariance = np.ones((self.DIM,), dtype=np.float16) # It seemed to us that float16 is enough (empirically)

        # Constant to calculate only once
        self._phi = norm.ppf(self._eta)

    def getModelSize(self):
        return self.mean.nbytes

        # Similarly to the convention of other papers, we consider only the size of the actual separators,
        # because they are the ones which will be used during test time.
        # The alternative is:
        # return self.mean.nbytes + self.covariance.nbytes

    # Refit sample x to predict label y.
    # The precalculated actual margin is passed to the method to save time
    def refit(self, x, y, actualMargin):
        # Check if the top negative code is not far enough from the lowest positive code
        if (y * actualMargin) <= self.margin:

            # Calculate m_i and v_i (The margin is already given in the parameters)
            # Not necessary when actual margin is given: m_i = np.dot(x.data, y).dot(self.mean[x.indices])
            m_i = y * actualMargin

            covG = np.multiply(x.data, self.covariance[x.indices])
            v_i = covG.dot(x.data)

            # Compute AROW's alpha beta
            beta_i = 1 / (v_i + self._r)
            alpha_i = max(1 - m_i, 0) * beta_i

            # Update means and covariance
            self.mean[x.indices] += alpha_i * y * covG
            self.covariance[x.indices] -= beta_i * np.power(covG, 2) # Taking only the diagonal

    def score(self, x):
        # The score is computed solely from the mean vectors
        return np.dot(x.data, self.mean[x.indices])

    def _prepareToTrain(self):
        # No special actions needed for AROW.
        pass

    def _prepareToEval(self):
        # No special actions needed for AROW.
        pass
