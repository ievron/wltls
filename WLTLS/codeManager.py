"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import numpy as np
import abc

from aux_lib import print_debug
from graphs.binaryTree import BinaryTree


#############################################################################################
# A bijective mapping between codewords and classes / labels.
# Explained in the paper under the "Path assignment" section.
#############################################################################################
class CodeManager(metaclass=abc.ABCMeta):
    _NO_LABEL = -1
    LABELS = None
    _bitsNum = None
    codingMatrix = None
    _codeIdxToLabel = None
    _labelToCodeIdx = None

    def __init__(self, LABELS, allCodes):
        allCodes = np.array(list(allCodes))

        self.LABELS = LABELS
        self._bitsNum = allCodes.shape[1]

        self.mapping = BinaryTree()
        self._buildCodingMatrix(allCodes)
        self._initMappings()

        print_debug("Using a {} path assignment policy.".format(self.getName()))

    # Get the column vector of a specific bit in the coding matrix
    def getCodingBit(self, col):
        return self.codingMatrix[:, col]

    @staticmethod
    @abc.abstractmethod
    def getName():
        raise NotImplementedError

    @abc.abstractmethod
    def _initMappings(self):
        raise NotImplementedError

    def _buildCodingMatrix(self, allCodes):
        # This helps computations during training only
        self.codingMatrix = np.zeros((self.LABELS, self._bitsNum), dtype=np.int8)

        for idx, code in enumerate(allCodes):
            self.codingMatrix[idx, :] = code

            self.mapping.store(code, idx)

    def codeToCodeIdx(self, code):
        return self.mapping.read(code)

    # Maps a code to its label
    # Optionally receives the real label, so its can send it to the assignment policy manager
    def codeToLabel(self, code, actualLabel = None):
        codeIdx = self.codeToCodeIdx(code)

        label = self._codeIdxToLabel[codeIdx]

        # If the code is still unassigned
        if label == self._NO_LABEL:
            return None

        return label

    def _mapCodeToLabel(self, label, codeIdx):
        self._labelToCodeIdx[label] = codeIdx
        self._codeIdxToLabel[codeIdx] = label

    def labelToCode(self, label):
        codeIdx = self._labelToCodeIdx[label]

        if codeIdx is None:
            return None

        return self.codingMatrix[codeIdx, :]

    # Assign any unassigned codes
    def assignRemainingCodes(self):
        # Find remaining labels
        remainingLabels = [label for label,code in enumerate(self._labelToCodeIdx) if code is None]

        i = 0
        # Iterate remaining codes
        for codeIdx, label in enumerate(self._codeIdxToLabel):
            if label == self._NO_LABEL:
                self._mapCodeToLabel(remainingLabels.pop(), codeIdx)
                i += 1

        return i

    # Assigns an available code to a label
    @abc.abstractmethod
    def assignLabel(self, label, potentialCodes):
        raise NotImplementedError

    def getFirstAvailableCodeIdx(self):
        return np.argmax(self._codeIdxToLabel == self._NO_LABEL)



#############################################################################################
# Randomly assigns codewords to classes.
#############################################################################################
class RandomCodeManager(CodeManager):
    def __init__(self, LABELS, allCodes):
        super().__init__(LABELS, allCodes)

    def _initMappings(self):
        self._codeIdxToLabel = [None] * self.LABELS
        self._labelToCodeIdx = [None] * self.LABELS

        prmt = np.random.permutation(self.LABELS)

        for idx, i in enumerate(prmt):
            self._codeIdxToLabel[idx] = i
            self._labelToCodeIdx[i] = idx

    def assignLabel(self, label, potentialCodes):
        raise RuntimeError("Shouldn't have reached here")

    @staticmethod
    def getName():
        return "Random"


#############################################################################################
# Assigns a newly-seen label to the first available code from a list of potential codes.
# Idea taken from [Log-time and Log-space Extreme Classification; Jasinska et al. 2016]
#############################################################################################
class GreedyCodeManager(CodeManager):
    def __init__(self, LABELS, allCodes):
        super().__init__(LABELS, allCodes)

    def _initMappings(self):
        self._codeIdxToLabel = np.ones((self.LABELS,), dtype=np.int) * self._NO_LABEL
        self._labelToCodeIdx = [None] * self.LABELS

    @staticmethod
    def getName():
        return "Greedy"

    def assignLabel(self, label, potentialCodes):
        for code in potentialCodes:
            assignedLabel = self.codeToLabel(code)

            # Check if the code is free to be assigned
            if assignedLabel is None:
                codeIdx = self.codeToCodeIdx(code)
                self._mapCodeToLabel(label, codeIdx)

                return code

        # If there was no available code among the potential ones, take the first available
        codeIdx = self.getFirstAvailableCodeIdx()
        self._mapCodeToLabel(label, codeIdx)

        return self.codingMatrix[codeIdx]


#############################################################################################
# Assigns codewords to labels by a given mapping.
# The i-th label in labelPermutation will belong to the i-th codeword.
#############################################################################################
class FixedCodeManager(CodeManager):
    def __init__(self, LABELS, allCodes, labelPermutation):
        self.labelPermutation = labelPermutation

        super().__init__(LABELS, allCodes)

    @staticmethod
    def getName():
        return "Fixed"

    def _initMappings(self):
        self._codeIdxToLabel = [None] * self.LABELS
        self._labelToCodeIdx = [None] * self.LABELS

        prmt = self.labelPermutation

        for idx, i in enumerate(prmt):
            self._codeIdxToLabel[idx] = i
            self._labelToCodeIdx[i] = idx

    def assignLabel(self, label, potentialCodes):
        raise RuntimeError("Shouldn't have reached here")
