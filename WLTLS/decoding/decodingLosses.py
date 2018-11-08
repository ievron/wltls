"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

import numpy as np

#########################################################################
# y is an array containing the actual {-1, 1} code of *one* sample.
# z is an array containing the soft margins of that sample.
# See the paper for more details
#########################################################################

# Squared hinge loss, most suitable for learning with AROW
def squaredHingeLoss(y, z):
    return np.power(np.maximum(1 - np.multiply(y, z), 0), 2) # Good for vectorize calculations!

# Squared loss, equivalent of the loss in [Log-time and Log-space Extreme Classification; Jasinska et al. 2016]
def squaredLoss(y, z):
    return np.power((1-np.multiply(y,z)), 2)

# Exponential loss. Practically worked best in most experiments.
def expLoss(y, z):
    # We truncate the powers for numerical reasons
    exp_ = np.clip(np.multiply(y,z), -20, 20)
    return np.exp(-exp_)