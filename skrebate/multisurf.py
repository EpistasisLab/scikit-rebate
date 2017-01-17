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
from .surfstar import SURFstar
from joblib import Parallel, delayed

class MultiSURF(SURFstar):

    """Feature selection using data-mined expert knowledge.

    Based on the MultiSURF algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """
############################# MultiSURF ########################################
    def _find_neighbors(self, inst):
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

    def _compute_scores(self, inst, attr, nan_entries):
        scores = np.zeros(self._num_attributes)
        NN_near, NN_far = self._find_neighbors(inst)

        for feature_num in range(self._num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += self._compute_score(attr, NN_near, feature_num, inst, nan_entries)
            if len(NN_far) > 0:
                scores[feature_num] -= self._compute_score(attr, NN_far, feature_num, inst, nan_entries)

        return scores

    def _run_algorithm(self):
        attr = self._get_attribute_info()
        nan_entries = np.isnan(self._X)
        
        if self.n_jobs != 1:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(self._compute_scores)(instance_num, attr, nan_entries) for instance_num in range(self._datalen)), axis=0)
        else:
            scores = np.sum([self._compute_scores(instance_num, attr, nan_entries) for instance_num in range(self._datalen)], axis=0)
        
        return np.array(scores)
