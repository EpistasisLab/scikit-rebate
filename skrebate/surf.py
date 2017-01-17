# -*- coding: utf-8 -*-

"""
Copyright (c) 2016 Randal S. Olson, Pete Schmitt, and Ryan J. Urbanowicz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import numpy as np
from joblib import Parallel, delayed
from .relieff import ReliefF

class SURF(ReliefF):

    """Feature selection using data-mined expert knowledge.

    Based on the SURF algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """
    def __init__(self, n_features_to_select=10, discrete_threshold=10, verbose=False, n_jobs=1):
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        n_features_to_select: int (default: 10)
            the number of top features (according to the relieff score) to 
            retain after feature selection is applied.
        discrete_threshold: int (default: 10)
            Value used to determine if a feature is discrete or continuous.
            If the number of unique levels in a feature is > discrete_threshold, then it is
            considered continuous, or discrete otherwise.
        verbose: bool (default: False)
            If True, output timing of distance array and scoring
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.

        """
        self.n_features_to_select = n_features_to_select
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs

############################# SURF ############################################
    def _find_neighbors(self, inst, avg_dist):
        NN = []
        min_indicies = []

        for i in range(self._datalen):
            if inst != i:
                locator = [inst,i]
                if i > inst:
                    locator.reverse()
                d = self._distance_array[locator[0]][locator[1]]
                if d < avg_dist:
                    min_indicies.append(i)
        for i in range(len(min_indicies)):
            NN.append(min_indicies[i])
        return np.array(NN, dtype=np.int32)

    def _compute_scores(self, inst, attr, nan_entries, avg_dist):
        scores = np.zeros(self._num_attributes)
        NN = self._find_neighbors(inst, avg_dist)
        if len(NN) <= 0:
            return scores
        for feature_num in range(self._num_attributes):
            scores[feature_num] += self._compute_score(attr, NN, feature_num, inst, nan_entries)
        return scores

    def _run_algorithm(self):
        sm = cnt = 0
        for i in range(self._datalen):
            sm += sum(self._distance_array[i])
            cnt += len(self._distance_array[i])
        avg_dist = sm / float(cnt)

        attr = self._get_attribute_info()
        nan_entries = np.isnan(self._X)

        if self.n_jobs != 1:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(self._compute_scores)(instance_num, attr, nan_entries, avg_dist) for instance_num in range(self._datalen)), axis=0)
        else:
            scores = np.sum([self._compute_scores(instance_num, attr, nan_entries, avg_dist) for instance_num in range(self._datalen)], axis=0)

        return np.array(scores)
