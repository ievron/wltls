"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
--------------------------------------------------------------------------------
See the paper for more details on the different losses.
"""

import numpy as np
import scipy.sparse as sps

from aux_lib import Timing

STOP_DELTA = 1.0  # Delta allowed for accuracy degradation
THRESHOLDS = 10  # Amount of thresholds
MIN_PERCENTILE = 1  # Minimum nnz to test

#########################################################################
# Experiment sparse models of a previously trained W-LTLS model.
#########################################################################
class SparseExperiment:
    def __init__(self, codeManager, decoder):
        self.codeManager = codeManager
        self.decoder = decoder


    def test(self, sparseMat, X, Y):
        correct = 0
        total = Y.shape[0]
        t = Timing()

        for x, y in zip(X, Y):
            w = x.dot(sparseMat).todense().A.ravel()

            code = self.decoder.findKBestCodes(w, 1)[0]

            if y == self.codeManager.codeToLabel(code):
                correct += 1

        return {
            "accuracy": correct * 100 / total,
            "time": t.get_elapsed_time()
        }

    def run(self, M, Xtest, Ytest, Xvalid, Yvalid):
        A = abs(M)

        # Calculate #THRESHOLDS thresholds to test from the current nonzeros to MIN_PERCENTILE nonzeros
        firstNnz = 100 - (np.count_nonzero(A) * 100 / (A.shape[0] * A.shape[1]))
        startFrom = firstNnz + ((100 - firstNnz) * 2 / 3)
        nonzeroPercents = [firstNnz] + list(np.linspace(startFrom, (100 - MIN_PERCENTILE), THRESHOLDS - 1))
        percentiles = [np.percentile(A, percent) for percent in nonzeroPercents]

        del A

        originalWeight = M.nbytes / (1024 ** 2)
        print("Original (unpruned) matrix weight: {:.1f}MB".format(originalWeight))

        print("  # | {:<9} | {:<9} | {:<15} | {:<14} |".format(
            "threshold", "Non-zeros", "Sparse mat. weight", "Validation acc",
        ))

        history = []

        # Iterate the thresholds
        for i, percentile in enumerate(percentiles):
            # Iterate the columns of the weight matrix (number of binary functions, \ell in paper)
            for l in range(M.shape[1]):
                indices = (M[:, l] < percentile) & (M[:, l] > -percentile)

                M[indices, l] = 0

            # Create sparse matrix
            sparseM = sps.csr_matrix(M)

            # Get validation and test results
            res = self.test(sparseM, Xtest, Ytest)
            valRes = self.test(sparseM, Xvalid, Yvalid)

            # Compute the sparse matrix weight (MB)
            sparseWeight = (sparseM.data.nbytes + sparseM.indptr.nbytes + sparseM.indices.nbytes) / (1024 ** 2)

            # Calc percentage of nonzero weights
            nnz = 100 * sparseM.count_nonzero() / (M.shape[0] * M.shape[1])

            print(" {:>2} | {:>9.7f} | {:>8.2f}% | {:>6.1f}MB = {:>6.2f}% | {:>13.2f}% |".format(
                i + 1, percentile,
                nnz, sparseWeight, sparseWeight * 100 / originalWeight,
                valRes["accuracy"], res["accuracy"],
                res["time"]))

            del sparseM

            history.append({
                "nnz": nnz,
                "threshold": percentile,
                "sparseWeight": sparseWeight,
                "valAccuracy": valRes["accuracy"],
                "testAccuracy": res["accuracy"],
                "valTime": valRes["time"],
                "testTime": res["time"],
            })

            # Check if the validation accuracy degradation is too high
            if history[0]["valAccuracy"] - valRes["accuracy"] >= STOP_DELTA:
                print("Stopping condition reached (excess validation accuracy degradation)!")
                break

            best = i

        # Best model
        print("\nBest model within allowed validation accuracy degradation:")
        print("#{}. threshold={:9.7f}, nonzeros: {:>8.2f}%, test accuracy: {:.2f}%".format(
            best + 1, history[best]["threshold"], history[best]["nnz"], history[best]["testAccuracy"]))

        return history