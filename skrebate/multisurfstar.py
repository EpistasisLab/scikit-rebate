# -*- coding: utf-8 -*-

"""
scikit-rebate was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Pete Schmitt (pschmitt@upenn.edu)
    - Ryan J. Urbanowicz (ryanurb@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - and many more generous open source contributors

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
from .surfstar import SURFstar
from .scoring_utils import MultiSURFstar_compute_scores
from sklearn.externals.joblib import Parallel, delayed


class MultiSURFstar(SURFstar):

    """Feature selection using data-mined expert knowledge.

    Based on the MultiSURF algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """

############################# MultiSURF* ########################################
    def _find_neighbors(self, inst):
        """ Identify nearest as well as farthest hits and misses within radius defined by average distance and standard deviation of distances from target instanace.
        This works the same regardless of endpoint type. """
        dist_vect = []
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                dist_vect.append(self._distance_array[locator[0]][locator[1]])

        dist_vect = np.array(dist_vect)
        inst_avg_dist = np.average(dist_vect)
        inst_std = np.std(dist_vect) / 2.
        near_threshold = inst_avg_dist - inst_std
        far_threshold = inst_avg_dist + inst_std

        NN_near = []
        NN_far = []
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                if self._distance_array[locator[0]][locator[1]] < near_threshold:
                    NN_near.append(j)
                elif self._distance_array[locator[0]][locator[1]] > far_threshold:
                    NN_far.append(j)

        return np.array(NN_near), np.array(NN_far)

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield MultiSURF* scores. """
        nan_entries = np.isnan(self._X)

        NNlist = [self._find_neighbors(datalen) for datalen in range(self._datalen)]
        NN_near_list = [i[0] for i in NNlist]
        NN_far_list = [i[1] for i in NNlist]

        scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
            MultiSURFstar_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                          NN_near, NN_far, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
            for instance_num, NN_near, NN_far in zip(range(self._datalen), NN_near_list, NN_far_list)), axis=0)

        return np.array(scores)
