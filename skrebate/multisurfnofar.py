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

class MultiSURFNoFar(SURFstar):

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

        NN_near = []
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                if self._distance_array[locator[0]][locator[1]] < near_threshold:
                    NN_near.append(j)

        return np.array(NN_near)

    def _compute_scores(self, inst, attr, nan_entries):
        scores = np.zeros(self._num_attributes)
        NN_near = self._find_neighbors(inst)

        for feature_num in range(self._num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += self._compute_score_near(attr, NN_near, feature_num, inst, nan_entries)

        return scores

    def _run_algorithm(self):
        attr = self._get_attribute_info()
        nan_entries = np.isnan(self._X)
        
        if self.n_jobs != 1:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                self._compute_scores)(instance_num, attr, nan_entries) for instance_num in range(self._datalen)), axis=0)
        else:
            scores = np.sum([self._compute_scores(instance_num, attr, nan_entries) for instance_num in range(self._datalen)], axis=0)
        
        return np.array(scores)

    ###############################################################################
    def _compute_score_near(self, attr, NN, feature, inst, nan_entries):
        """Evaluates feature scores according to the ReliefF algorithm"""

        fname = self._headers[feature]
        ftype = attr[fname][0]  # feature type
        ctype = self._class_type # class type
        diff_hit = diff_miss = 0.0 
        count_hit = count_miss = 0.0
        mmdiff = 1
        diff = 0

        if nan_entries[inst][feature]:
            return 0.

        xinstfeature = self._X[inst][feature]

        #--------------------------------------------------------------------------
        if ctype == 'discrete':
            for i in range(len(NN)):
                if nan_entries[NN[i]][feature]:
                    continue

                xNNifeature = self._X[NN[i]][feature]
                absvalue = abs(xinstfeature - xNNifeature) / mmdiff
    
                if self._y[inst] == self._y[NN[i]]:   # HIT
                    count_hit += 1
                    if xinstfeature != xNNifeature:
                        if ftype == 'continuous':
                            diff_hit -= absvalue
                        else: # discrete
                            diff_hit -= 1
                else: # MISS
                    count_miss += 1
                    if xinstfeature != xNNifeature:
                        if ftype == 'continuous':
                            diff_miss += absvalue
                        else: # discrete
                            diff_miss += 1

            hit_proportion = count_hit / float(len(NN))
            miss_proportion = count_miss / float(len(NN))
            diff = diff_hit * miss_proportion + diff_miss * hit_proportion
        #--------------------------------------------------------------------------
        else: # CONTINUOUS endpoint
            mmdiff = attr[fname][3]
            same_class_bound = self._labels_std

            for i in range(len(NN)):
                if nan_entries[NN[i]][feature]:
                    continue

                xNNifeature = self._X[NN[i]][feature]
                absvalue = abs(xinstfeature - xNNifeature) / mmdiff

                if abs(self._y[inst] - self._y[NN[i]]) < same_class_bound: # HIT
                    count_hit += 1
                    if xinstfeature != xNNifeature:
                        if ftype == 'continuous':
                            diff_hit -= absvalue
                        else: # discrete
                            diff_hit -= 1
                else: # MISS
                    count_miss += 1
                    if xinstfeature != xNNifeature:
                        if ftype == 'continuous':
                            diff_miss += absvalue
                        else: # discrete
                            diff_miss += 1

            hit_proportion = count_hit / float(len(NN))
            miss_proportion = count_miss / float(len(NN))
            diff = diff_hit * miss_proportion + diff_miss * hit_proportion

        return diff
