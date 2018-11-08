"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

import abc

class Learner(metaclass=abc.ABCMeta):
    PREDICTORS = 0
    DIM = 0
    _isTrain = None

    def __init__(self, DIM, isTrain=True):
        self.DIM = DIM
        self._isTrain = isTrain

    @abc.abstractmethod
    def getModelSize(self):
        raise NotImplementedError

    def getPredictorsNumber(self):
        return self.PREDICTORS

    @abc.abstractmethod
    def refit(self, x, y, actualMargin):
        raise NotImplementedError

    # Predict many using the final model
    def predictFinal(self, X):
        raise NotImplementedError

    # Score the learners a sample
    @abc.abstractmethod
    def score(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def _prepareToTrain(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _prepareToEval(self):
        raise NotImplementedError

    # Switches the model to evaluation mode
    def eval(self):
        if self._isTrain:
            self._isTrain = False
            self._prepareToEval()


    # Switches the model to test mode
    def train(self):
        if not self._isTrain:
            self._isTrain = True
            self._prepareToTrain()
