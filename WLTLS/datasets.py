"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

import itertools
import numpy as np
import random
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from aux_lib import Timing

# Taken from [Log-time and Log-space Extreme Classification; Jasinska et al. 2016]
class LabelEncoder2():
    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        if self.multilabel:
            self.le = MultiLabelBinarizer(sparse_output=True)
        else:
            self.le = LabelEncoder()
        self.from_classes = False

    def fit(self, Y):
        self.le.fit(Y)
        self.from_classes = False

    def transform(self, Y):
        if self.from_classes:
            if self.multilabel:
                all_in_Y = set(itertools.chain.from_iterable(Y))
            else:
                all_in_Y = set()
                for el in np.unique(Y):
                    all_in_Y.add(el)
            self.new_classes = sorted(all_in_Y.difference(set(self.classes_)))
            self.num_in_training = len(self.classes_)
            self.num_new_in_test = len(self.new_classes)

            Y2 = []
            if self.multilabel:
                for yy in Y:
                    y2 = []
                    for y in yy:
                        if y in self.classes_:
                            y2.append(y)
                    Y2.append(y2)
            else:
                for y in Y:
                    if y in self.classes_:
                        Y2.append(y)
                    else:
                        # random class? or remove the example? or _no_class_ marker?
                        Y2.append(random.choice(self.classes_))
            Y = Y2

            self.le.classes_ = self.classes_
        return self.le.transform(Y)

    def inverse_transform(self, Y):
        Y = self.le.inverse_transform(Y)
        return Y

    def set_classes(self, classes_):
        self.classes_ = classes_
        self.from_classes = True

    def get_classes(self):
        return self.le.classes_

# Taken from [Log-time and Log-space Extreme Classification; Jasinska et al. 2016]
def load_dataset(path_train, path_valid, path_test, n_features, multilabel=False, classes_=None):
    le = LabelEncoder2(multilabel=multilabel)

    X, Y, Xvalid, Yvalid, Xtest, Ytest = load_svmlight_files((path_train, path_valid, path_test), dtype=np.float32,
                                                             n_features=n_features,
                                                             multilabel=multilabel)
    if classes_ is None:
        le.fit(np.concatenate((Y, Yvalid, Ytest), axis=0))
        Y = le.transform(Y)
        Yvalid = le.transform(Yvalid)
        Ytest = le.transform(Ytest)
    else:
        le.set_classes(classes_)
        Y = le.transform(Y)
        Yvalid = le.transform(Yvalid)
    return X, Y, Xvalid, Yvalid, Xtest, Ytest

def read(path, dataset, printSummary = True):
    t = Timing()

    specificPath = "{0}/{1}/{1}".format(path, dataset.name)

    sorted_extension = "_sorted" if dataset.name == "bibtex" else ""

    Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = \
        load_dataset(
            "{}.train{}".format(specificPath, sorted_extension),
            "{}.heldout{}".format(specificPath, sorted_extension),
            "{}.test{}".format(specificPath, sorted_extension),
            dataset.n_features,
            multilabel=dataset.multilabel)

    if dataset.multilabel:
        LABELS = len(set(list(Ytrain.indices) + list(Yvalid.indices) + list(Ytest.indices)))
    else:
        LABELS = len(set(list(Ytrain) + list(Yvalid) + list(Ytest)))

    DIMS = Xtrain.shape[1]

    # Print summary
    if printSummary:
        effectiveDim = Xtrain.nnz / Xtrain.shape[0]
        print(("{} dataset '{}':\n" +
              "\tLoaded in:\t{:}\n" +
              "\tLabels:\t\tK={:,}\n" +
               "\tFeatures:\td={:,} ({:.1f} non-zero features on average)").format(
            "Multi-label" if dataset.multilabel else "Multi-class",
            dataset.name,
            t.get_elapsed_time(), LABELS, DIMS, effectiveDim))

    return Xtrain, Ytrain.astype(np.int), Xvalid, Yvalid.astype(np.int), Xtest, Ytest.astype(np.int), LABELS, DIMS


class DatasetParams:
    def __init__(self, name, epochs, multilabel, n_features):
        if multilabel:
            raise NotImplementedError

        self.name = name
        self.epochs = epochs
        self.multilabel = multilabel
        self.n_features = n_features

class datasets:
    #                               name,               epochs,   multilabel,   n_features
    sector =        DatasetParams(  "sector",           5,        False,        55197)
    aloi_bin =      DatasetParams(  "aloi.bin",         8,        False,        636949)
    LSHTC1 =        DatasetParams(  "LSHTC1",           5,        False,        1199856)
    imageNet =      DatasetParams(  "imageNet",         1,        False,        1000)
    Dmoz =          DatasetParams(  "Dmoz",             7,        False,        833484)

    @staticmethod
    def getParams(datasetName):
        return getattr(datasets, datasetName.replace(".", "_"))

    @staticmethod
    def getAll():
        for a in dir(datasets):
            if not a.startswith('__') and not callable(getattr(datasets, a)):
                yield datasets.getParams(a)