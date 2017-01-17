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
from .surf import SURF

class SURFstar(SURF):

    """Feature selection using data-mined expert knowledge.

    Based on the SURF* algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """

############################# SURF* ########################################
    def _find_neighbors(self, inst, avg_dist):
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

    def _compute_scores(self, inst, attr, nan_entries, avg_dist):
        scores = np.zeros(self._num_attributes)
        NN_near, NN_far = self._find_neighbors(inst, avg_dist)
        for feature_num in range(self._num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += self._compute_score(attr, NN_near, feature_num, inst, nan_entries)
            if len(NN_far) > 0:
                scores[feature_num] -= self._compute_score(attr, NN_far, feature_num, inst, nan_entries)
        return scores
