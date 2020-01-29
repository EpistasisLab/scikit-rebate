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
from .surf import SURF
from .scoring_utils import SURFstar_compute_scores
from sklearn.externals.joblib import Parallel, delayed


class SURFstar(SURF):

    """Feature selection using data-mined expert knowledge.

    Based on the SURF* algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """

############################# SURF* ########################################
    def _find_neighbors(self, inst, avg_dist):
        """ Identify nearest as well as farthest hits and misses within radius defined by average distance over whole distance array.
        This works the same regardless of endpoint type. """
        NN_near = []
        NN_far = []
        min_indices = []
        max_indices = []

        for i in range(self._datalen):
            if inst != i:
                locator = [inst, i]
                if i > inst:
                    locator.reverse()
                d = self._distance_array[locator[0]][locator[1]]
                if d < avg_dist:
                    min_indices.append(i)
                if d > avg_dist:
                    max_indices.append(i)

        for i in range(len(min_indices)):
            NN_near.append(min_indices[i])
        for i in range(len(max_indices)):
            NN_far.append(max_indices[i])

        return np.array(NN_near, dtype=np.int32), np.array(NN_far, dtype=np.int32)

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield SURF* scores. """
        sm = cnt = 0
        for i in range(self._datalen):
            sm += sum(self._distance_array[i])
            cnt += len(self._distance_array[i])
        avg_dist = sm / float(cnt)

        nan_entries = np.isnan(self._X)

        NNlist = [self._find_neighbors(datalen, avg_dist) for datalen in range(self._datalen)]
        NN_near_list = [i[0] for i in NNlist]
        NN_far_list = [i[1] for i in NNlist]

        scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
            SURFstar_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                     NN_near, NN_far, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
            for instance_num, NN_near, NN_far in zip(range(self._datalen), NN_near_list, NN_far_list)), axis=0)

        return np.array(scores)
