"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import numpy as np

#############################################################################################
# A W-LTLS model.
#############################################################################################
class WltlsModel():
    POTENTIAL_PATH_NUM = 20
    LABELS = 0
    DIM = 0
    codeManager = None
    _learners = None
    _decoder = None

    def __init__(self, LABELS, DIM, learnerClass, codeManager, decoder, isMultilabel = False):
        if isMultilabel:
            raise NotImplementedError

        self.codeManager = codeManager
        learners = [learnerClass(DIM) for _ in range(self.codeManager._bitsNum)]

        self.LABELS = LABELS
        self.DIM = DIM
        self._learners = learners
        self._assignedAllLabels = False
        self._isMultilabel = isMultilabel

        self._decoder = decoder
        self.loss = decoder.loss
        print("Model size: {0:.1f}MB".format(self.getModelSize() / 1024 ** 2))

    def getActualLabels(self, y):
        return y.indices if self._isMultilabel else [y]

    def getModelSize(self):
        return sum([l.getModelSize() for l in self._learners])

    def getPredictorsNumber(self):
        return len(self._learners)

    # If we predict a yet-assigned code, we return None
    def _predictAndRefit(self, x, y):
        # Currently works only for multiclass! Multilabel require significant adjustments, see LTLS

        actualLabel = self.getActualLabels(y)[0]
        margins = self._getMargins(x)

        # It's a step towards multi-label support, but it doesn't work in this version.
        # In multi-class, there's only one positive (actual) label, and at most one negative (predicted) label
        actualCode = self.codeManager.labelToCode(actualLabel)

        # If the actual label isn't assigned yet to a path, find POTENTIAL_PATH_NUM heaviest paths to choose from
        if actualCode is None:
            actualCode = self.codeManager.assignLabel(actualLabel,
                                                      self._decoder.findKBestCodes(margins, self.POTENTIAL_PATH_NUM))


        # Learn independently
        #
        # Note about parallel learning:
        # This part could be made parallel with |learners| cores.
        # However, the code should be organized a little differently (the outer loop should be on cores/learners
        # and the inner one on the samples).
        # Moreover, to support the greedy path assignment method (see paper), we have to make a prediction for every
        # sample sequentially. The random path assignment policy fully support the learning in parallel.
        for i,l in enumerate(self._learners):
            y = actualCode[i]

            l.refit(x, y, margins[i])


        # Find best code (decoding)
        topCode = self._decoder.findKBestCodes(margins, 1)[0]

        # Find predicted label
        yPredicted = self.codeManager.codeToLabel(topCode)

        return yPredicted

    def train(self, X, Y):
        # Switches the decoding to test mode
        for l in self._learners:
            l.train()

        # Actually train (refit) the learners
        # (This part could be made parallel with |learners| cores)
        yPredicted = [self._predictAndRefit(x, y) for x,y in zip(X, Y)]

        return yPredicted

    # Predicts a sample's label.
    def _predict(self, x, k=1):
        margins = self._getMargins(x)

        codes = self._decoder.findKBestCodes(margins, k)

        return [self.codeManager.codeToLabel(code) for code in codes]

    # Predicts a sample's label.
    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    # Compute the values of the binary functions on an input x
    def _getMargins(self, x):
        return [l.score(x) for l in self._learners]

    # Prepare for evaluation
    def eval(self):
        extraOutput = None

        # Before the first evaluation, assign all remaining codewords to paths.
        # Important in cases some class appear on the validation/test set but not on the train set.
        if not self._assignedAllLabels:
            extraOutput = self.prepareForFirstEvaluation()

            self._assignedAllLabels = True

        for l in self._learners:
            l.eval()

        return extraOutput

    # Predict many using the final model (after training)
    def predictFinal(self, X):
        extraOutput = self.eval()

        yPredicted = [self._predict(x) for x in X]

        return yPredicted, extraOutput

    # Called after the first training epoch, before the first evaluation.
    # Should return output to be printed, or None.
    def prepareForFirstEvaluation(self):
        # Assign codes to unassigned labels
        assigned = self.codeManager.assignRemainingCodes()

        return "Assigned {} labels.".format(assigned)