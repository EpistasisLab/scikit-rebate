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
import sys
from .surfstar import SURFstar
from .scoring_utils import SWRFstar_compute_scores
from joblib import Parallel, delayed


class SWRFstar(SURFstar):

    """

    Based on the SWRF* algorithm as introduced in:

    Matthew E. Stokes and Shyam Visweswaran Application of a spatially-weighted
    Relief algorithm for ranking genetic predictors of disease.
    
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
                #find out what locator means, check out line 200 of ReliefF
                if d < avg_dist:
                    min_indices.append(i)
                if d > avg_dist:
                    max_indices.append(i)
                

        for i in range(len(min_indices)):
            NN_near.append(min_indices[i])
        for i in range(len(max_indices)):
            NN_far.append(max_indices[i])

        # Make a vector of distances between target instance (inst) and all others
        dist_vect = []
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                dist_vect.append(self._distance_array[locator[0]][locator[1]])

        dist_vect = np.array(dist_vect)


        return np.array(NN_near, dtype=np.int32), np.array(NN_far, dtype=np.int32), dist_vect

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield SWRF* scores. """
        sm = cnt = 0
        for i in range(self._datalen):
            sm += sum(self._distance_array[i])
            cnt += len(self._distance_array[i])
        avg_dist = sm / float(cnt)
        #check line 227 of ReliefF

        nan_entries = np.isnan(self._X)

        NNlist = [self._find_neighbors(datalen, avg_dist) for datalen in range(self._datalen)]
        NN_near_list = [i[0] for i in NNlist]
        NN_far_list = [i[1] for i in NNlist]
        dist_vectors = [i[2] for i in NNlist]

        #for SWRF and SWRF* only, avg_dist is passed as a parameter as it is used in the sigmoid calculations
        if isinstance(self._weights,np.ndarray) and self.weight_final_scores:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                SWRFstar_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                         NN_near, NN_far, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type, avg_dist, self._distance_array,self._weights)
                for instance_num, NN_near, NN_far in zip(range(self._datalen), NN_near_list, NN_far_list)), axis=0)

        else:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                SWRFstar_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                         NN_near, NN_far, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type, avg_dist, self._distance_array)
                for instance_num, NN_near, NN_far in zip(range(self._datalen), NN_near_list, NN_far_list)), axis=0)

        return np.array(scores)

