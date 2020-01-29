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
from .scoring_utils import get_row_missing, ReliefF_compute_scores


class ReliefF(BaseEstimator):

    """Feature selection using data-mined expert knowledge.
    Based on the ReliefF algorithm as introduced in:
    Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55"""

    """Note that ReliefF class establishes core functionality that is inherited by all other Relief-based algorithms.
    Assumes: * There are no missing values in the label/outcome/dependent variable.
             * For ReliefF, the setting of k is <= to the number of instances that have the least frequent class label
             (binary and multiclass endpoint data. """

    def __init__(self, n_features_to_select=10, n_neighbors=100, discrete_threshold=10, verbose=False, n_jobs=1):
        """Sets up ReliefF to perform feature selection. Note that an approximation of the original 'Relief'
        algorithm may be run by setting 'n_features_to_select' to 1. Also note that the original Relief parameter 'm'
        is not included in this software. 'm' specifies the number of random training instances out of 'n' (total
        training instances) used to update feature scores. Since scores are most representative when m=n, all
        available training instances are utilized in all Relief-based algorithm score updates here. If the user
        wishes to utilize a smaller 'm' in Relief-based scoring, simply pass any of these algorithms a subset of the
        original training dataset samples.

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
        """Scikit-learn required: Computes the feature importance scores from the training data.

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
        self._X = X  # matrix of predictive variables ('independent variables')
        self._y = y  # vector of values for outcome variable ('dependent variable')

        # Set up the properties for ReliefF -------------------------------------------------------------------------------------
        self._datalen = len(self._X)  # Number of training instances ('n')

        """"Below: Handles special case where user requests that a proportion of training instances be neighbors for
        ReliefF rather than a specified 'k' number of neighbors.  Note that if k is specified, then k 'hits' and k
        'misses' will be used to update feature scores.  Thus total number of neighbors is 2k. If instead a proportion
        is specified (say 0.1 out of 1000 instances) this represents the total number of neighbors (e.g. 100). In this
        case, k would be set to 50 (i.e. 50 hits and 50 misses). """
        if hasattr(self, 'n_neighbors') and type(self.n_neighbors) is float:
            # Halve the number of neighbors because ReliefF uses n_neighbors matches and n_neighbors misses
            self.n_neighbors = int(self.n_neighbors * self._datalen * 0.5)

        # Number of unique outcome (label) values (used to determine outcome variable type)
        self._label_list = list(set(self._y))
        # Determine if label is discrete
        discrete_label = (len(self._label_list) <= self.discrete_threshold)

        # Identify label type (binary, multiclass, or continuous)
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

        self._num_attributes = len(self._X[0])  # Number of features in training data
        
        # Number of missing data values in predictor variable matrix.
        self._missing_data_count = np.isnan(self._X).sum()

        """Assign internal headers for the features (scikit-learn does not accept external headers from dataset):
        The pre_normalize() function relies on the headers being ordered, e.g., X01, X02, etc.
        If this is changed, then the sort in the pre_normalize() function needs to be adapted as well. """
        xlen = len(self._X[0])
        mxlen = len(str(xlen + 1))
        self._headers = ['X{}'.format(str(i).zfill(mxlen)) for i in range(1, xlen + 1)]

        start = time.time()  # Runtime tracking

        # Determine data types for all features/attributes in training data (i.e. discrete or continuous)
        C = D = False
        # Examines each feature and applies discrete_threshold to determine variable type.
        self.attr = self._get_attribute_info()
        for key in self.attr.keys():
            if self.attr[key][0] == 'discrete':
                D = True
            if self.attr[key][0] == 'continuous':
                C = True

        # For downstream computational efficiency, determine if dataset is comprised of all discrete, all continuous, or a mix of discrete/continuous features.
        if C and D:
            self.data_type = 'mixed'
        elif D and not C:
            self.data_type = 'discrete'
        elif C and not D:
            self.data_type = 'continuous'
        else:
            raise ValueError('Invalid data type in data set.')
        #--------------------------------------------------------------------------------------------------------------------

        # Compute the distance array between all data points ----------------------------------------------------------------
        # For downstream efficiency, separate features in dataset by type (i.e. discrete/continuous)
        diffs, cidx, didx = self._dtype_array()
        cdiffs = diffs[cidx]  # max/min continuous value difference for continuous features.

        xc = self._X[:, cidx]  # Subset of continuous-valued feature data
        xd = self._X[:, didx]  # Subset of discrete-valued feature data

        """ For efficiency, the distance array is computed more efficiently for data with no missing values.
        This distance array will only be used to identify nearest neighbors. """
        if self._missing_data_count > 0:
            self._distance_array = self._distarray_missing(xc, xd, cdiffs)
        else:
            self._distance_array = self._distarray_no_missing(xc, xd)

        if self.verbose:
            elapsed = time.time() - start
            print('Created distance array in {} seconds.'.format(elapsed))
            print('Feature scoring under way ...')

        start = time.time()
       #--------------------------------------------------------------------------------------------------------------------

       # Run remainder of algorithm (i.e. identification of 'neighbors' for each instance, and feature scoring).------------
        # Stores feature importance scores for ReliefF or respective Relief-based algorithm.
        self.feature_importances_ = self._run_algorithm()

        # Delete the internal distance array because it is no longer needed
        del self._distance_array

        if self.verbose:
            elapsed = time.time() - start
            print('Completed scoring in {} seconds.'.format(elapsed))

        # Compute indices of top features
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        return self

    #=========================================================================#
    def transform(self, X):
        """Scikit-learn required: Reduces the feature set down to the top `n_features_to_select` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_select}
            Reduced feature matrix

        """
        if self._num_attributes < self.n_features_to_select:
            raise ValueError('Number of features to select is larger than the number of features in the dataset.')
        
        return X[:, self.top_features_[:self.n_features_to_select]]

    #=========================================================================#
    def fit_transform(self, X, y):
        """Scikit-learn required: Computes the feature importance scores from the training data, then reduces the feature set down to the top `n_features_to_select` features.

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
        """ Relief algorithms handle the scoring updates a little differently for data with multiclass outcomes. In ReBATE we implement multiclass scoring in line with
        the strategy described by Kononenko 1994 within the RELIEF-F variant which was suggested to outperform the RELIEF-E multiclass variant. This strategy weights
        score updates derived from misses of different classes by the class frequency observed in the training data. 'The idea is that the algorithm should estimate the
        ability of attributes to separate each pair of classes regardless of which two classes are closest to each other'.  In this method we prepare for this normalization
        by creating a class dictionary, and storing respective class frequencies. This is needed for ReliefF multiclass score update normalizations. """
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
        """ Preprocess the training dataset to identify which features/attributes are discrete vs. continuous valued. Ignores missing values in this determination."""
        attr = dict()
        d = 0
        limit = self.discrete_threshold
        w = self._X.transpose()

        for idx in range(len(w)):
            h = self._headers[idx]
            z = w[idx]
            if self._missing_data_count > 0:
                z = z[np.logical_not(np.isnan(z))]  # Exclude any missing values from consideration
            zlen = len(np.unique(z))
            if zlen <= limit:
                attr[h] = ('discrete', 0, 0, 0, 0)
                d += 1
            else:
                mx = np.max(z)
                mn = np.min(z)
                sd = np.std(z)
                attr[h] = ('continuous', mx, mn, mx - mn, sd)
        # For each feature/attribute we store (type, max value, min value, max min difference, average, standard deviation) - the latter three values are set to zero if feature is discrete.
        return attr

    def _distarray_no_missing(self, xc, xd):
        """Distance array calculation for data with no missing values. The 'pdist() function outputs a condense distance array, and squareform() converts this vector-form
        distance vector to a square-form, redundant distance matrix.
        *This could be a target for saving memory in the future, by not needing to expand to the redundant square-form matrix. """
        from scipy.spatial.distance import pdist, squareform

        #------------------------------------------#
        def pre_normalize(x):
            """Normalizes continuous features so they are in the same range (0 to 1)"""
            idx = 0
            # goes through all named features (doesn really need to) this method is only applied to continuous features
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

        if self.data_type == 'discrete':  # discrete features only
            return squareform(pdist(self._X, metric='hamming'))
        elif self.data_type == 'mixed':  # mix of discrete and continuous features
            d_dist = squareform(pdist(xd, metric='hamming'))
            # Cityblock is also known as Manhattan distance
            c_dist = squareform(pdist(pre_normalize(xc), metric='cityblock'))
            return np.add(d_dist, c_dist) / self._num_attributes

        else: #continuous features only
            #xc = pre_normalize(xc)
            return squareform(pdist(pre_normalize(xc), metric='cityblock'))

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
        """Distance array calculation for data with missing values"""
        cindices = []
        dindices = []
        # Get Boolean mask locating missing values for continuous and discrete features separately. These correspond to xc and xd respectively.
        for i in range(self._datalen):
            cindices.append(np.where(np.isnan(xc[i]))[0])
            dindices.append(np.where(np.isnan(xd[i]))[0])

        if self.n_jobs != 1:
            dist_array = Parallel(n_jobs=self.n_jobs)(delayed(get_row_missing)(
                xc, xd, cdiffs, index, cindices, dindices) for index in range(self._datalen))
        else:
            # For each instance calculate distance from all other instances (in non-redundant manner) (i.e. computes triangle, and puts zeros in for rest to form square).
            dist_array = [get_row_missing(xc, xd, cdiffs, index, cindices, dindices)
                          for index in range(self._datalen)]

        return np.array(dist_array)
    #==================================================================#

############################# ReliefF ############################################

    def _find_neighbors(self, inst):
        """ Identify k nearest hits and k nearest misses for given instance. This is accomplished differently based on the type of endpoint (i.e. binary, multiclass, and continuous). """
        # Make a vector of distances between target instance (inst) and all others
        dist_vect = []
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                dist_vect.append(self._distance_array[locator[0]][locator[1]])
            else:
                # Ensures that target instance is never selected as neighbor.
                dist_vect.append(sys.maxsize)

        dist_vect = np.array(dist_vect)

        # Identify neighbors-------------------------------------------------------
        """ NN for Binary Endpoints: """
        if self._class_type == 'binary':
            nn_list = []
            match_count = 0
            miss_count = 0
            for nn_index in np.argsort(dist_vect):
                if self._y[inst] == self._y[nn_index]:  # Hit neighbor identified
                    if match_count >= self.n_neighbors:
                        continue
                    nn_list.append(nn_index)
                    match_count += 1
                else:  # Miss neighbor identified
                    if miss_count >= self.n_neighbors:
                        continue
                    nn_list.append(nn_index)
                    miss_count += 1

                if match_count >= self.n_neighbors and miss_count >= self.n_neighbors:
                    break

        elif self._class_type == 'multiclass':
            nn_list = []
            match_count = 0
            miss_count = dict.fromkeys(self._label_list, 0)
            for nn_index in np.argsort(dist_vect):
                if self._y[inst] == self._y[nn_index]:  # Hit neighbor identified
                    if match_count >= self.n_neighbors:
                        continue
                    nn_list.append(nn_index)
                    match_count += 1
                else:
                    for label in self._label_list:
                        if self._y[nn_index] == label:
                            if miss_count[label] >= self.n_neighbors:
                                continue
                            nn_list.append(nn_index)
                            miss_count[label] += 1

                if match_count >= self.n_neighbors and all(v >= self.n_neighbors for v in miss_count.values()):
                    break
        else:
            nn_list = []
            match_count = 0
            miss_count = 0
            for nn_index in np.argsort(dist_vect):
                if abs(self._y[inst]-self._y[nn_index]) < self._labels_std:  # Hit neighbor identified
                    if match_count >= self.n_neighbors:
                        continue
                    nn_list.append(nn_index)
                    match_count += 1
                else:  # Miss neighbor identified
                    if miss_count >= self.n_neighbors:
                        continue
                    nn_list.append(nn_index)
                    miss_count += 1

                if match_count >= self.n_neighbors and miss_count >= self.n_neighbors:
                    break
        return np.array(nn_list)

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield ReliefF scores. """

        # Find nearest neighbors
        NNlist = map(self._find_neighbors, range(self._datalen))

        # Feature scoring - using identified nearest neighbors
        nan_entries = np.isnan(self._X)  # boolean mask for missing data values

        # Call the scoring method for the ReliefF algorithm
        scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
            ReliefF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                    NN, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
            for instance_num, NN in zip(range(self._datalen), NNlist)), axis=0)

        return np.array(scores)
