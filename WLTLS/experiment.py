"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

import numpy as np
from aux_lib import Timing
import random

#########################################################################
# An entire experiment.
# Train and test model. Check binary losses.
#########################################################################
class Experiment():
    _model = None
    EPOCHS = None
    trainAccuracies = None
    validAccuracies = None
    trainTimes = None

    def __init__(self, model, epochs = 1):
        self._model = model
        self.EPOCHS = epochs
        self.trainAccuracies = np.zeros((self.EPOCHS,), dtype=np.float16)
        self.validAccuracies = np.zeros((self.EPOCHS,), dtype=np.float16)
        self.trainTimes = np.zeros((self.EPOCHS,), dtype=np.int16)

    # Test the model on a given set
    def _test(self, X, Y):
        t = Timing()

        yPredicted, extraOutput = self._model.predictFinal(X)

        accuracy = 100 * self.evaluateScore(Y, yPredicted)

        return { "accuracy": accuracy,
                 "time": t.get_elapsed_secs(),
                 "y_predicted": yPredicted,
                 "extra_output": extraOutput }


    # Return accuracy
    @staticmethod
    def evaluateScore(Y, predictedY):
        correct = 0

        for actual, predicted in zip(Y, predictedY):
            if actual == predicted:
                correct += 1

        return correct / len(Y)

    # Shuffle the given sets
    def _shuffleDataset(self, X, Y):
        c = list(zip(X, Y))

        random.shuffle(c)

        return zip(*c)

    def _runEpoch(self, id, Xtrain, Ytrain, Xvalid, Yvalid):
        Xtrain, Ytrain = self._shuffleDataset(Xtrain, Ytrain)

        print("Train epoch {}/{}: ".format(id + 1, self.EPOCHS), end='', flush=True)

        t = Timing()
        t.start()

        yPredicted = self._model.train(Xtrain, Ytrain)

        trainAccuracy = 100 * self.evaluateScore(Ytrain, yPredicted)

        self.trainAccuracies[id] = trainAccuracy
        self.trainTimes[id] = t.get_elapsed_secs()
        print("{:.1f}% in {}.\t".format(
            trainAccuracy, t.get_elapsed_time()), end='', flush=True)

        # Validation
        validRes = self._test(Xvalid, Yvalid)
        self.validAccuracies[id] = validRes["accuracy"]
        print('Validation: {:.1f}% ({}).\t'.format(validRes["accuracy"], Timing.secondsToString(validRes["time"])), end='', flush=True)

        if id == 0 and validRes["extra_output"] is not None:
            print(validRes["extra_output"], end=' ', flush=True)

        return validRes

    def run(self, Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest, modelLogPath=None,
            returnBestValidatedModel = False):
        bestRes = 0

        if modelLogPath is not None:
            # If we need to save the model, we increase the recursion limit
            # so we can save objects like the tree in the code manager etc.
            # The code itself is not recursive!
            import sys
            sys.setrecursionlimit(10 ** 5)

        # Run training epochs
        for epoch in range(0, self.EPOCHS):
            validRes = self._runEpoch(epoch, Xtrain, Ytrain, Xvalid, Yvalid)

            # If our validation score is the highest so far
            if validRes["accuracy"] > bestRes:
                bestRes = validRes["accuracy"]

                # We save the model to file whenever we reach a better validation performance,
                # so that if the simulation will be terminated for some reason
                # (usually only happen when we didn't allocate enough space for the process)
                # we will have a backup.
                if modelLogPath is not None:
                    tSave = Timing()
                    tSave.start()

                    # Important:
                    # numpy.savez is originally meant for saving arrays, not objects,
                    # We use it here for simplicity but it sometimes causes very high additional memory requirements
                    # (it processes the data before saving).
                    #
                    # A better way would be to save the data of the binary learners (e.g. means of AROW),
                    # and the coding matrix and allocation, one by one, and then load them with a special method.
                    np.savez(modelLogPath, ltls=self._model)
                    print("Saved model ({}).".format(tSave.get_elapsed_time()), end='', flush=True)
                    del tSave

            print("")

        # If we need to return the best validated model, load it from file before continuing
        if returnBestValidatedModel:
            del self._model

            self._model = np.load(modelLogPath + ".npz")["ltls"][()]

        # Test
        testRes = self._test(Xtest, Ytest)
        print('Test accuracy: {:.1f}% ({})'.format(testRes["accuracy"], Timing.secondsToString(testRes["time"])))

        # Calculate average binary loss
        decodingLoss = self._calcBitwiseLoss(Xtrain, Ytrain)
        print("Average binary loss: {:.2f}".format(decodingLoss))



    # Calculate the average binary loss (\epsilon in the paper)
    def _calcBitwiseLoss(self, X, Y):
        lossFunc = self._model.loss

        loss = np.array([lossFunc(self._model._getMargins(x),
                                  self._model.codeManager.labelToCode(y))
                         for x,y in zip(X,Y)])

        return np.mean(loss)