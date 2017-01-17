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
import sys
import math
import time as tm
import warnings
from numpy import isnan, where, append, unique, delete, empty, double, array
from numpy import std, subtract, logical_not, max, min, sum, absolute
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

class ReliefF(BaseEstimator):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """
    def __init__(self, n_features_to_select=10, n_neighbors=100, discrete_threshold=10, verbose=False, n_jobs=1):
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        n_features_to_select: int (default: 10)
            the number of top features (according to the relieff score) to 
            retain after feature selection is applied.
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature
            importance scores. More neighbors results in more accurate scores,
            but takes longer.
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
        self.n_neighbors = n_neighbors
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs

    #=========================================================================#
    def fit(self, X, y):
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        
        Returns
        -------
        Copy of the ReliefF instance

        """
        self._X = X
        self._y = y

        # Disallow parallelization in Python 2
        if self.n_jobs != 1 and sys.version_info[0] < 3:
            warnings.warn('Parallelization is currently not supported in Python 2. Settings n_jobs to 1.', RuntimeWarning)
            self.n_jobs = 1

        # Set up the properties for ReliefF
        self._datalen = len(self._X)
        self._label_list = list(set(self._y))
        discrete_label = (len(self._label_list) <= self.discrete_threshold)

        if discrete_label:
            self._class_type = 'discrete'
        else:
            self._class_type = 'continuous'

        # Training labels standard deviation -- only used if the training labels are continuous
        self._labels_std = 0.
        if len(self._label_list) > self.discrete_threshold:
            self._labels_std = std(self._y, ddof=1)

        self._num_attributes = len(self._X[0])
        self._missing_data_count = isnan(self._X).sum()

        # Assign internal headers for the features
        xlen = len(self._X[0])
        mxlen = len(str(xlen + 1))
        self._headers = ['X{}'.format(str(i).zfill(mxlen)) for i in range(1, xlen + 1)]
        
        # Determine the data type
        C = D = False
        attr = self._get_attribute_info()
        for key in attr.keys():
            if attr[key][0] == 'discrete':
                D = True
            if attr[key][0] == 'continuous':
                C = True

        if C and D: 
            self.data_type = 'mixed'
        elif D and not C:
            self.data_type = 'discrete'
        elif C and not D:
            self.data_type = 'continuous'
        else:
            raise ValueError('Invalid data type in data set.')

        # Compute the distance array between all data points
        start = tm.time()

        attr = self._get_attribute_info()
        diffs, cidx, didx = self._dtype_array(attr)
        cdiffs = diffs[cidx]
        xc = self._X[:,cidx]
        xd = self._X[:,didx]

        if self._missing_data_count > 0:
            self._distance_array = self._distarray_missing(xc, xd, cdiffs)
        else:
            self._distance_array = self._distarray_no_missing(xc, xd)

        if self.verbose:
            elapsed = tm.time() - start
            print('Created distance array in {} seconds.'.format(elapsed))
            print('Feature scoring under way ...')

        start = tm.time()
        self.feature_importances_ = self._run_algorithm()

        if self.verbose:
            elapsed = tm.time() - start
            print('Completed scoring in {} seconds.'.format(elapsed))

        # Compute indices of top features
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        # Delete the internal distance array because it is no longer needed
        del self._distance_array

        return self

    #=========================================================================#
    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_select` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_select}
            Reduced feature matrix

        """
        return X[:, self.top_features_[:self.n_features_to_select]]

    #=========================================================================#
    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_select` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_select}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X)

######################### SUPPORTING FUNCTIONS ###########################
    def _get_attribute_info(self):
        attr = dict()
        d = 0
        limit = self.discrete_threshold
        w = self._X.transpose()
        
        for idx in range(len(w)):
            h = self._headers[idx]
            z = w[idx]
            if self._missing_data_count > 0:
                z = z[logical_not(isnan(z))]
            zlen = len(unique(z)) 
            if zlen <= limit:
                attr[h] = ('discrete', 0, 0, 0)
                d += 1
            else:
                mx = max(z)
                mn = min(z)
                attr[h] = ('continuous', mx, mn, mx - mn)
        
        return attr
    #==================================================================#    
    def _distarray_no_missing(self, xc, xd):
        """ distance array for data with no missing values """
        from scipy.spatial.distance import pdist, squareform
        attr = self._get_attribute_info()
        #------------------------------------------#
        def pre_normalize(x):
            idx = 0
            for i in attr:
                cmin = attr[i][2]
                diff = attr[i][3]
                x[idx] -= cmin
                x[idx] /= diff
                idx += 1
            return x
        #------------------------------------------#
        if self.data_type == 'discrete':
            return squareform(pdist(self._X, metric='hamming'))
        elif self.data_type == 'mixed':
            d_dist = squareform(pdist(xd, metric='hamming'))
            c_dist = squareform(pdist(pre_normalize(xc), metric='cityblock'))
            return np.add(d_dist, c_dist) / self._num_attributes
        else:
            self._X = pre_normalize(self._X)
            return squareform(pdist(self._X, metric='cityblock'))

    #==================================================================#
    def _dtype_array(self, attr):
        """  Return mask for discrete(0)/continuous(1) attributes and their 
             indices. Return array of max/min diffs of attributes. """
        attrtype = []
        attrdiff = []
        
        for key in self._headers:
            if attr[key][0] == 'continuous':
                attrtype.append(1)
            else:
                attrtype.append(0)
            attrdiff.append(attr[key][3])
            
        attrtype = array(attrtype)
        cidx = where(attrtype == 1)[0]
        didx = where(attrtype == 0)[0]
        
        attrdiff = array(attrdiff)
        return attrdiff, cidx, didx
    #==================================================================#
    def _distarray_missing(self, xc, xd, cdiffs):
        """ distance array for data with missing values """
        cindices = []
        dindices = []
        for i in range(self._datalen):
            cindices.append(where(isnan(xc[i]))[0])
            dindices.append(where(isnan(xd[i]))[0])
    
        dist_array = Parallel(n_jobs=self.n_jobs)(delayed(self._get_row_missing)(xc, xd, cdiffs, index, cindices, dindices) for index in range(self._datalen))
        return np.array(dist_array)
    #==================================================================#
    def _get_row_missing(self, xc, xd, cdiffs, index, cindices, dindices):
        row = empty(0, dtype=double)
        cinst1 = xc[index]
        dinst1 = xd[index]
        can = cindices[index]
        dan = dindices[index]
        for j in range(index):
            dist = 0
            dinst2 = xd[j]
            cinst2 = xc[j]

            # continuous
            cbn = cindices[j]
            idx = unique(append(can, cbn))   # create unique list
            c1 = delete(cinst1, idx)       # remove elements by idx
            c2 = delete(cinst2, idx)
            cdf = delete(cdiffs, idx)

            # discrete
            dbn = dindices[j]
            idx = unique(append(dan, dbn))
            d1 = delete(dinst1, idx)
            d2 = delete(dinst2, idx)
            
            # discrete first
            dist += len(d1[d1 != d2])

            # now continuous
            dist += sum(absolute(subtract(c1, c2)) / cdf)

            row = append(row, dist)
        return row
    
############################# ReliefF ############################################
    def _find_neighbors(self, inst):
        dist_vect = []
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                dist_vect.append(self._distance_array[locator[0]][locator[1]])
            else:
                dist_vect.append(0.)

        dist_vect = np.array(dist_vect)
        return np.argsort(dist_vect)[1:self.n_neighbors + 1]

    def _compute_scores(self, inst, attr, nan_entries):
        scores = np.zeros(self._num_attributes)
        NN = self._find_neighbors(inst)
        for feature_num in range(self._num_attributes):
            scores[feature_num] += self._compute_score(attr, NN, feature_num, inst, nan_entries)
        return scores

    def _run_algorithm(self):
        attr = self._get_attribute_info()
        nan_entries = isnan(self._X)
        
        if self.n_jobs != 1:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(self._compute_scores)(instance_num, attr, nan_entries) for instance_num in range(self._datalen)), axis=0)
        else:
            scores = np.sum([self._compute_scores(instance_num, attr, nan_entries) for instance_num in range(self._datalen)], axis=0)

        return np.array(scores)

    ###############################################################################
    def _compute_score(self, attr, NN, feature, inst, nan_entries):
        """ evaluates ReliefF scores """

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
