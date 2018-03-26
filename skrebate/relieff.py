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
import time
import warnings
import sys
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed
from scoring_utils import get_row_missing, ReliefF_compute_scores


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
        n_neighbors: int or float (default: 100)
            The number of neighbors to consider when assigning feature
            importance scores. If a float number is provided, that percentage of
            training samples is used as the number of neighbors.
            More neighbors results in more accurate scores, but takes longer.
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
        # Set up the properties for ReliefF
        self._datalen = len(self._X)
        if hasattr(self, 'n_neighbors') and type(self.n_neighbors) is float:
            # Halve the number of neighbors because ReliefF uses n_neighbors matches
            # and n_neighbors misses
            self.n_neighbors = int(self.n_neighbors * self._datalen * 0.5)
        self._label_list = list(set(self._y))
        discrete_label = (len(self._label_list) <= self.discrete_threshold)

        if discrete_label:
            if len(self._label_list) == 2:
                self._class_type = 'binary'
                self.mcmap = 0
            elif len(self._label_list) > 2:
                self._class_type = 'multiclass'
                self.mcmap = self._getMultiClassMap()
            else:
                raise ValueError('All labels are of the same class.')

        else:
            self._class_type = 'continuous'
            self.mcmap = 0

        # Training labels standard deviation -- only used if the training labels are continuous
        self._labels_std = 0.
        if len(self._label_list) > self.discrete_threshold:
            self._labels_std = np.std(self._y, ddof=1)

        self._num_attributes = len(self._X[0])
        self._missing_data_count = np.isnan(self._X).sum()

        # Assign internal headers for the features
        # The pre_normalize() function relies on the headers being ordered, e.g., X01, X02, etc.
        # If this is changed, then the sort in the pre_normalize() function needs to be adapted as well.
        xlen = len(self._X[0])
        mxlen = len(str(xlen + 1))
        self._headers = ['X{}'.format(str(i).zfill(mxlen)) for i in range(1, xlen + 1)]

        start = time.time()
        # Determine the data type
        C = D = False
        self.attr = self._get_attribute_info()
        for key in self.attr.keys():
            if self.attr[key][0] == 'discrete':
                D = True
            if self.attr[key][0] == 'continuous':
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

        diffs, cidx, didx = self._dtype_array()
        cdiffs = diffs[cidx]
        xc = self._X[:, cidx]
        xd = self._X[:, didx]

        if self._missing_data_count > 0:
            self._distance_array = self._distarray_missing(xc, xd, cdiffs)
        else:
            self._distance_array = self._distarray_no_missing(xc, xd)

        if self.verbose:
            elapsed = time.time() - start
            print('Created distance array in {} seconds.'.format(elapsed))
            print('Feature scoring under way ...')

        start = time.time()
        self.feature_importances_ = self._run_algorithm()

        if self.verbose:
            elapsed = time.time() - start
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
        """Computes the feature importance scores from the training data, then reduces the feature set down to the top `n_features_to_select` features.

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
    def _getMultiClassMap(self):

        mcmap = dict()

        for i in range(self._datalen):
            if(self._y[i] not in mcmap):
                mcmap[self._y[i]] = 0
            else:
                mcmap[self._y[i]] += 1

        for each in self._label_list:
            mcmap[each] = mcmap[each]/float(self._datalen)

        return mcmap

    def _get_attribute_info(self):
        attr = dict()
        d = 0
        limit = self.discrete_threshold
        w = self._X.transpose()

        for idx in range(len(w)):
            h = self._headers[idx]
            z = w[idx]
            if self._missing_data_count > 0:
                z = z[np.logical_not(np.isnan(z))]
            zlen = len(np.unique(z))
            if zlen <= limit:
                attr[h] = ('discrete', 0, 0, 0)
                d += 1
            else:
                mx = np.max(z)
                mn = np.min(z)
                attr[h] = ('continuous', mx, mn, mx - mn)

        return attr
    #==================================================================#

    def _distarray_no_missing(self, xc, xd):
        """Distance array for data with no missing values"""
        from scipy.spatial.distance import pdist, squareform

        #------------------------------------------#

        def pre_normalize(x):
            """Normalizes continuous features so they are in the same range"""
            idx = 0
            for i in sorted(self.attr.keys()):
                if self.attr[i][0] == 'discrete':
                    continue
                cmin = self.attr[i][2]
                diff = self.attr[i][3]
                x[:, idx] -= cmin
                x[:, idx] /= diff
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
    def _dtype_array(self):
        """Return mask for discrete(0)/continuous(1) attributes and their indices. Return array of max/min diffs of attributes."""
        attrtype = []
        attrdiff = []

        for key in self._headers:
            if self.attr[key][0] == 'continuous':
                attrtype.append(1)
            else:
                attrtype.append(0)
            attrdiff.append(self.attr[key][3])

        attrtype = np.array(attrtype)
        cidx = np.where(attrtype == 1)[0]
        didx = np.where(attrtype == 0)[0]

        attrdiff = np.array(attrdiff)
        return attrdiff, cidx, didx
    #==================================================================#

    def _distarray_missing(self, xc, xd, cdiffs):
        """Distance array for data with missing values"""
        cindices = []
        dindices = []
        for i in range(self._datalen):
            cindices.append(np.where(np.isnan(xc[i]))[0])
            dindices.append(np.where(np.isnan(xd[i]))[0])

        if self.n_jobs != 1:
            dist_array = Parallel(n_jobs=self.n_jobs)(delayed(get_row_missing)(
                xc, xd, cdiffs, index, cindices, dindices) for index in range(self._datalen))
        else:
            dist_array = [get_row_missing(xc, xd, cdiffs, index, cindices, dindices)
                          for index in range(self._datalen)]

        return np.array(dist_array)
    #==================================================================#
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
                dist_vect.append(sys.maxsize)

        dist_vect = np.array(dist_vect)

        nn_list = []
        match_count = 0
        miss_count = 0
        for nn_index in np.argsort(dist_vect):
            if self._y[inst] == self._y[nn_index]:  # match
                if match_count >= self.n_neighbors:
                    continue
                nn_list.append(nn_index)
                match_count += 1
            else:  # miss
                if miss_count >= self.n_neighbors:
                    continue
                nn_list.append(nn_index)
                miss_count += 1

            if match_count >= self.n_neighbors and miss_count >= self.n_neighbors:
                break

        return np.array(nn_list)

    def _run_algorithm(self):

        nan_entries = np.isnan(self._X)

        NNlist = map(self._find_neighbors, range(self._datalen))
        scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
            ReliefF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                    NN, self._headers, self._class_type, self._X, self._y, self._labels_std)
            for instance_num, NN in zip(range(self._datalen), NNlist)), axis=0)

        return np.array(scores)
