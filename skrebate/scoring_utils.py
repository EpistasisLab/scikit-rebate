# -*- coding: utf-8 -*-

"""
scikit-rebate was primarily developed at the University of Pennsylvania and at the Cedars-Sinai Health Sciences University by:
    - Ryan J. Urbanowicz (ryanurb@upenn.edu)
    - Randal S. Olson (rso@randalolson.com)
    - Pete Schmitt (pschmitt@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Ting-Hui Wu (tinghui333w@gmail.com)
    - Kia Kazemi-Nia (kia.kazemi-nia@cshs.org)
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

import numpy as np


# (Subset of continuous-valued feature data, Subset of discrete-valued (categorical) feature data, max/min difference, instance index, boolean mask for continuous, boolean mask for discrete)
def get_row_missing(xc, xd, cdiffs, index, cindices, dindices):
    """ Calculate distance between index instance and all other instances. """
    # row = np.empty(0, dtype=np.double)  # initialize empty row
    # initialize a full‐length row of zeros (distance to self and future indices stays 0)
    n = xc.shape[0]
    row = np.zeros(n, dtype=np.double)
    cinst1 = xc[index]  # continuous-valued features for index instance
    dinst1 = xd[index]  # discrete-valued features for index instance
    # Boolean mask locating missing values for continuous features for index instance
    can = cindices[index]
    # Boolean mask locating missing values for discrete features for index instance
    dan = dindices[index]
    tf = len(cinst1) + len(dinst1)  # total number of features.

    # Progressively compare current instance to all others. Excludes comparison with self indexed instance. (Building the distance matrix triangle).
    for j in range(index):
        dist = 0
        dinst2 = xd[j]  # discrete-valued features for compared instance
        cinst2 = xc[j]  # continuous-valued features for compared instance

        # Manage missing values in discrete features
        # Boolean mask locating missing values for discrete features for compared instance
        dbn = dindices[j]
        # indexes where there is at least one missing value in the feature between an instance pair.
        idx = np.unique(np.append(dan, dbn))
        # Number of features excluded from distance calculation due to one or two missing values within instance pair. Used to normalize distance values for comparison.
        dmc = len(idx)
        d1 = np.delete(dinst1, idx)  # delete unique missing features from index instance
        d2 = np.delete(dinst2, idx)  # delete unique missing features from compared instance
        # print("D1:", d1)
        # print("D2:", d2, "\n")

        # Manage missing values in continuous features
        # Boolean mask locating missing values for continuous features for compared instance
        cbn = cindices[j]
        # indexes where there is at least one missing value in the feature between an instance pair.
        idx = np.unique(np.append(can, cbn))
        # Number of features excluded from distance calculation due to one or two missing values within instance pair. Used to normalize distance values for comparison.
        cmc = len(idx)
        c1 = np.delete(cinst1, idx)  # delete unique missing features from index instance
        c2 = np.delete(cinst2, idx)  # delete unique missing features from compared instance
        # delete unique missing features from continuous value difference scores
        cdf = np.delete(cdiffs, idx)

        # Add discrete feature distance contributions (missing values excluded) - Hamming distance
        dist += len(d1[d1 != d2])
        # print("Hamming distance:", dist)

        # Add continuous feature distance contributions (missing values excluded) - Manhattan distance (Note that 0-1 continuous value normalization is included ~ subtraction of minimums cancel out)
        dist += np.sum(np.absolute(np.subtract(c1, c2)) / cdf)

        # Normalize distance calculation based on total number of missing values bypassed in either discrete or continuous features.
        tnmc = tf - dmc - cmc  # Total number of unique missing counted
        # Distance normalized by number of features included in distance sum (this seeks to handle missing values neutrally in distance calculation)
        dist = dist/float(tnmc)
        # print("Hamming distance after division by tnmc:", dist)

        # row = np.append(row, dist)
        # place into the pre‐allocated slot
        row[j] = dist

    return row

# For iter relief
def get_row_missing_iter(xc, xd, cdiffs, index, cindices, dindices, weights):
    """ Calculate distance between index instance and all other instances. """
    # row = np.empty(0, dtype=np.double)  # initialize empty row
    # initialize a full‐length row of zeros (distance to self and future indices stays 0)
    n = xc.shape[0]
    row = np.zeros(n, dtype=np.double)
    cinst1 = xc[index]  # continuous-valued features for index instance
    dinst1 = xd[index]  # discrete-valued features for index instance
    # Boolean mask locating missing values for continuous features for index instance
    can = cindices[index]
    # Boolean mask locating missing values for discrete features for index instance
    dan = dindices[index]
    tf = len(cinst1) + len(dinst1)  # total number of features.
    # Progressively compare current instance to all others. Excludes comparison with self indexed instance. (Building the distance matrix triangle).
    for j in range(index):
        dist = 0
        dinst2 = xd[j]  # discrete-valued features for compared instance
        cinst2 = xc[j]  # continuous-valued features for compared instance

        # Manage missing values in discrete features
        # Boolean mask locating missing values for discrete features for compared instance
        dbn = dindices[j]
        # indexes where there is at least one missing value in the feature between an instance pair.
        idx = np.unique(np.append(dan, dbn))
        # Number of features excluded from distance calculation due to one or two missing values within instance pair. Used to normalize distance values for comparison.
        dmc = len(idx)
        d1 = np.delete(dinst1, idx)  # delete unique missing features from index instance
        d2 = np.delete(dinst2, idx)  # delete unique missing features from compared instance

        wd = np.delete(weights, idx)  # delete weights corresponding to missing discrete features
        # Manage missing values in continuous features
        # Boolean mask locating missing values for continuous features for compared instance
        cbn = cindices[j]
        # indexes where there is at least one missing value in the feature between an instance pair.
        idx = np.unique(np.append(can, cbn))
        # Number of features excluded from distance calculation due to one or two missing values within instance pair. Used to normalize distance values for comparison.
        cmc = len(idx)
        c1 = np.delete(cinst1, idx)  # delete unique missing features from index instance
        c2 = np.delete(cinst2, idx)  # delete unique missing features from compared instance
        # delete unique missing features from continuous value difference scores
        cdf = np.delete(cdiffs, idx)
        wc = np.delete(weights, idx)  # delete weights corresponding to missing continuous features

        # Add discrete feature distance contributions (missing values excluded) - Hamming distance
        if len(d1)!=0: #To ensure there is atleast one discrete variable
            hamming_dist = np.not_equal(d1, d2).astype(float)
            weight_hamming_dist = np.dot(hamming_dist, wd)/np.sum(wd)
            dist += weight_hamming_dist

        # Add continuous feature distance contributions (missing values excluded) - Manhattan distance (Note that 0-1 continuous value normalization is included ~ subtraction of minimums cancel out)
        if len(c1)!=0: #To ensure there is atleast one continuous variable
            dist += np.dot((np.absolute(np.subtract(c1, c2)) / cdf), wc)/np.sum(wc)

        # Normalize distance calculation based on total number of missing values bypassed in either discrete or continuous features.
        tnmc = tf - dmc - cmc  # Total number of unique missing counted
        # Distance normalized by number of features included in distance sum (this seeks to handle missing values neutrally in distance calculation)
        dist = dist/float(tnmc)

        # row = np.append(row, dist)
        # place into the pre‐allocated slot
        row[j] = dist

    return row

def ramp_vec(data_type, attr, fname, xinstfeature, x_nn):
    """
    Vectorized ramp function.
    Our own user simplified variation of the ramp function suggested by Hong 1994, 1997. Hong's method requires the user to specifiy two thresholds
    that indicate the max difference before a score of 1 is given, as well a min difference before a score of 0 is given, and any in the middle get a
    score that is the normalized difference between the two continuous feature values. This was done because when discrete and continuous features were mixed,
    continuous feature scores were underestimated.  Towards simplicity, automation, and a dataset adaptable approach,
    here we simply check whether the difference is greater than the standard deviation for the given feature; if so we assign a score of 1, otherwise we
    assign the normalized feature score difference.  This should help compensate for the underestimation.
    xinstfeature : scalar
    x_nn         : NumPy array
    returns      : NumPy array of diffs
    """
    mmdiff = attr[fname][3]   # max-min range of values for target feature
    rawfd = np.abs(xinstfeature - x_nn) # prenormalized feature value difference

    if data_type == 'mixed':
        standDev = attr[fname][4]

        # If rawfd > standDev → 1, else rawfd / mmdiff
        diff = np.where(
            rawfd > standDev,
            1.0,
            rawfd / mmdiff
        )
    else:
        diff = rawfd / mmdiff

    return diff

def compute_score(attr, mcmap, NN, feature, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, near=True):
    """Flexible feature scoring method that can be used with any core Relief-based method. Scoring proceeds differently
    based on whether endpoint is binary, multiclass, or continuous. This method is called for a single target instance
    + feature combination and runs over all items in NN. """

    fname = headers[feature]  # feature identifier
    ftype = attr[fname][0]  # feature type
    ctype = class_type  # class type (binary, multiclass, continuous)
    diff_hit = diff_miss = 0.0  # Tracks the score contribution
    # Tracks the number of hits/misses. Used in normalizing scores by 'k' in ReliefF, and by m or h in SURF, SURF*, MultiSURF*, and MultiSURF
    count_hit = count_miss = 0.0
    # Initialize 'diff' (The score contribution for this target instance and feature over all NN)
    diff = 0
    # mmdiff = attr[fname][3] # Max/Min range of values for target feature

    datalen = float(len(X))

    # If target instance is missing, then a 'neutral' score contribution of 0 is returned immediately since all NN comparisons will be against this missing value.
    if nan_entries[inst][feature]:
        return 0.
    # Note missing data normalization below regarding missing NN feature values is accomplished by counting hits and misses (missing values are not counted) (happens in parallel with hit/miss imbalance normalization)

    xinstfeature = X[inst][feature]  # value of target instances target feature.

    # NEW: bringing y_inst out of the loop and just looking it up once
    y_inst = y[inst]
    #--------------------------------------------------------------------------
    if ctype == 'binary':
        if near:
            # Step 1: Filter neighbors with non-missing feature values
            valid = ~nan_entries[NN, feature]
            nn_valid = NN[valid]

            if nn_valid.size == 0:
                count_hit = 0
                count_miss = 0
                diff_hit = 0.0
                diff_miss = 0.0
            else:
                # Step 2: Gather neighbor feature values & labels
                x_nn = X[nn_valid, feature]
                y_nn = y[nn_valid]

                # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                hits = (y_nn == y_inst)
                misses = ~hits

                count_hit = hits.sum()
                count_miss = misses.sum()

                # Step 4: Score updates
                if ftype == 'continuous':
                    # vectorized ramp function
                    # ramp_vec must accept arrays
                    diff_hit -= ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[hits]).sum()
                    diff_miss += ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[misses]).sum()
                else:
                    # discrete feature
                    diff_hit -= np.sum(x_nn[hits] != xinstfeature)
                    diff_miss += np.sum(x_nn[misses] != xinstfeature)
        else: # Far scoring
            # Step 1: Filter neighbors with non-missing feature values
            valid = ~nan_entries[NN, feature]
            nn_valid = NN[valid]

            if nn_valid.size == 0:
                count_hit = 0
                count_miss = 0
                diff_hit = 0.0
                diff_miss = 0.0
            else:
                # Step 2: Gather neighbor feature values & labels
                x_nn = X[nn_valid, feature]
                y_nn = y[nn_valid]

                # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                hits = (y_nn == y_inst)
                misses = ~hits

                count_hit = hits.sum()
                count_miss = misses.sum()

                # Step 4: Score updates
                if ftype == 'continuous':
                    # vectorized ramp function
                    # ramp_vec must accept arrays
                    diff_hit -= (count_hit - ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[hits]).sum())
                    diff_miss += (count_miss - ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[misses]).sum())
                else:
                    # discrete feature
                    diff_hit -= np.sum(x_nn[hits] == xinstfeature)
                    diff_miss += np.sum(x_nn[misses] == xinstfeature)

        """ Score Normalizations:
        *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
        *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)"""
        if count_hit == 0.0 or count_miss == 0.0:  # Special case, avoid division error
            if count_hit == 0.0 and count_miss == 0.0:
                return 0.0
            elif count_hit == 0.0:
                diff = (diff_miss / count_miss) / datalen
            else:  # count_miss == 0.0
                diff = (diff_hit / count_hit) / datalen
        else:  # Normal diff normalization
            diff = ((diff_hit / count_hit) + (diff_miss / count_miss)) / datalen

    #--------------------------------------------------------------------------
    elif ctype == 'multiclass':
        # only 'miss' classes will be stored
        class_store = {
            cls: [0, 0]
            for cls in mcmap
            if cls != y[inst]
        }

        if near:
            # Step 1: Filter neighbors with non-missing feature values
            valid = ~nan_entries[NN, feature]
            nn_valid = NN[valid]

            if nn_valid.size == 0:
                # No need to set count_miss and diff_miss for each miss class to 0 because they are already initialized to 0 in class_store
                count_hit = 0
                diff_hit = 0.0
            else:
                # Step 2: Gather neighbor feature values & labels
                x_nn = X[nn_valid, feature]
                y_nn = y[nn_valid]

                # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                hits = (y_nn == y_inst)
                count_hit = hits.sum()
                misses = ~hits

                # Step 4: Score updates
                if ftype == 'continuous':
                    # vectorized ramp function
                    # ramp_vec must accept arrays
                    diff_hit -= ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[hits]).sum()
                    
                    miss_diffs = ramp_vec(data_type, attr, fname, xinstfeature, x_nn[misses])
                    classes, inv = np.unique(y_nn[misses], return_inverse=True)

                    counts = np.bincount(inv)
                    diff_sums = np.bincount(inv, weights=miss_diffs)
                    for cls, cnt, diff_sum in zip(classes, counts, diff_sums):
                        class_store[cls][0] = cnt
                        class_store[cls][1] = diff_sum

                else:
                    # discrete feature
                    diff_hit -= np.sum(x_nn[hits] != xinstfeature)
                    
                    miss_diffs = (x_nn[misses] != xinstfeature).astype(float)
                    classes, inv = np.unique(y_nn[misses], return_inverse=True)

                    counts = np.bincount(inv)
                    diff_sums = np.bincount(inv, weights=miss_diffs)
                    for cls, cnt, diff_sum in zip(classes, counts, diff_sums):
                        class_store[cls][0] = cnt
                        class_store[cls][1] = diff_sum

        else: # Far scoring
            # Step 1: Filter neighbors with non-missing feature values
            valid = ~nan_entries[NN, feature]
            nn_valid = NN[valid]

            if nn_valid.size == 0:
                # No need to set count_miss and diff_miss for each miss class to 0 because they are already initialized to 0 in class_store
                count_hit = 0
                diff_hit = 0.0
            else:
                # Step 2: Gather neighbor feature values & labels
                x_nn = X[nn_valid, feature]
                y_nn = y[nn_valid]

                # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                hits = (y_nn == y_inst)
                count_hit = hits.sum()
                misses = ~hits

                # Step 4: Score updates
                if ftype == 'continuous':
                    # vectorized ramp function
                    # ramp_vec must accept arrays
                    diff_hit -= (count_hit - ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[hits]).sum())
                    
                    miss_diffs = ramp_vec(data_type, attr, fname, xinstfeature, x_nn[misses])
                    classes, inv = np.unique(y_nn[misses], return_inverse=True)

                    counts = np.bincount(inv)
                    diff_sums = np.bincount(inv, weights=miss_diffs)
                    for cls, cnt, diff_sum in zip(classes, counts, diff_sums):
                        class_store[cls][0] = cnt
                        class_store[cls][1] = (cnt - diff_sum)
                else:
                    # discrete feature
                    diff_hit -= np.sum(x_nn[hits] == xinstfeature)
                    
                    miss_diffs = (x_nn[misses] == xinstfeature).astype(float)
                    classes, inv = np.unique(y_nn[misses], return_inverse=True)

                    counts = np.bincount(inv)
                    diff_sums = np.bincount(inv, weights=miss_diffs)
                    for cls, cnt, diff_sum in zip(classes, counts, diff_sums):
                        class_store[cls][0] = cnt
                        class_store[cls][1] = diff_sum

        """ Score Normalizations:
        *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
        *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)
        * multiclass normalization - accounts for scoring by multiple miss class, so miss scores don't have too much weight in contrast with hit scoring. If a given miss class isn't included in NN
        then this normalization will account for that possibility. """
        # Miss component
        for each in class_store:
            count_miss += class_store[each][0]

        if count_hit == 0.0 and count_miss == 0.0:
            return 0.0
        else:
            if count_miss == 0:
                pass
            else: #Normal diff normalization
                for each in class_store: #multiclass normalization
                    diff += class_store[each][1] * (class_store[each][0] / count_miss) * len(class_store)# Contribution of given miss class weighted by it's observed frequency within NN set.
                diff = diff / count_miss #'m' normalization
            
            #Hit component: with 'h' normalization
            if count_hit == 0:
                pass
            else:
                diff += (diff_hit / count_hit)

        diff = diff / datalen  # 'n' normalization

    #--------------------------------------------------------------------------
    else:  # CONTINUOUS endpoint
        same_class_bound = labels_std

        if near:
            # Step 1: Filter neighbors with non-missing feature values
            valid = ~nan_entries[NN, feature]
            nn_valid = NN[valid]

            if nn_valid.size == 0:
                count_hit = 0
                count_miss = 0
                diff_hit = 0.0
                diff_miss = 0.0
            else:
                # Step 2: Gather neighbor feature values & labels
                x_nn = X[nn_valid, feature]
                y_nn = y[nn_valid]

                # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                hits = (np.abs(y_inst - y_nn) < same_class_bound)
                misses = ~hits

                count_hit = hits.sum()
                count_miss = misses.sum()

                # Step 4: Score updates
                if ftype == 'continuous':
                    # vectorized ramp function
                    # ramp_vec must accept arrays
                    diff_hit -= ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[hits]).sum()
                    diff_miss += ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[misses]).sum()
                else:
                    # discrete feature
                    diff_hit -= np.sum(x_nn[hits] != xinstfeature)
                    diff_miss += np.sum(x_nn[misses] != xinstfeature)
        else: # Far scoring
            # Step 1: Filter neighbors with non-missing feature values
            valid = ~nan_entries[NN, feature]
            nn_valid = NN[valid]

            if nn_valid.size == 0:
                count_hit = 0
                count_miss = 0
                diff_hit = 0.0
                diff_miss = 0.0
            else:
                # Step 2: Gather neighbor feature values & labels
                x_nn = X[nn_valid, feature]
                y_nn = y[nn_valid]

                # Step 3: Identify hits vs misses; hits and misses are boolean arrays through elementwise comparison
                hits = (np.abs(y_inst - y_nn) < same_class_bound)
                misses = ~hits

                count_hit = hits.sum()
                count_miss = misses.sum()

                # Step 4: Score updates
                if ftype == 'continuous':
                    # vectorized ramp function
                    # ramp_vec must accept arrays
                    diff_hit -= (count_hit - ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[hits]).sum())
                    diff_miss += (count_miss - ramp_vec(data_type, attr, fname,
                                        xinstfeature, x_nn[misses]).sum())
                else:
                    # discrete feature
                    diff_hit -= np.sum(x_nn[hits] == xinstfeature)
                    diff_miss += np.sum(x_nn[misses] == xinstfeature)

        """ Score Normalizations:
        *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
        *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)"""

        if count_hit == 0.0 or count_miss == 0.0:  # Special case, avoid division error
            if count_hit == 0.0 and count_miss == 0.0:
                return 0.0
            elif count_hit == 0.0:
                diff = (diff_miss / count_miss) / datalen
            else:  # count_miss == 0.0
                diff = (diff_hit / count_hit) / datalen
        else:  # Normal diff normalization
            diff = ((diff_hit / count_hit) + (diff_miss / count_miss)) / datalen

    return diff

def ReliefF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN, headers, class_type, X, y, labels_std, data_type, weights=None):
    """ Unique scoring procedure for ReliefF algorithm. Scoring based on k nearest hits and misses of current target instance. """
    scores = np.zeros(num_attributes)
    if isinstance(weights, np.ndarray):
        for feature_num in range(num_attributes):
            scores[feature_num] += weights[feature_num] * compute_score(attr, mcmap, NN, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
    else:
        for feature_num in range(num_attributes):
            scores[feature_num] += compute_score(attr, mcmap, NN, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
    return scores

def SURF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN, headers, class_type, X, y, labels_std, data_type, weights=None):
    """ Unique scoring procedure for SURF algorithm. Scoring based on nearest neighbors within defined radius of current target instance. """
    scores = np.zeros(num_attributes)
    if isinstance(weights, np.ndarray):
        if len(NN) <= 0:
            return scores
        for feature_num in range(num_attributes):
            scores[feature_num] += weights[feature_num] * compute_score(attr, mcmap, NN, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
    else:
        if len(NN) <= 0:
            return scores
        for feature_num in range(num_attributes):
            scores[feature_num] += compute_score(attr, mcmap, NN, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
    return scores


def SURFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std, data_type, weights=None):
    """ Unique scoring procedure for SURFstar algorithm. Scoring based on nearest neighbors within defined radius, as well as
    'anti-scoring' of far instances outside of radius of current target instance"""
    scores = np.zeros(num_attributes)
    if isinstance(weights, np.ndarray):
        for feature_num in range(num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += weights[feature_num] * compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
            # Note that we are using the near scoring loop in 'compute_score' and then just subtracting it here, in line with original SURF* paper.
            if len(NN_far) > 0:
                scores[feature_num] -= weights[feature_num] * compute_score(attr, mcmap, NN_far, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
    else:
        for feature_num in range(num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
            # Note that we are using the near scoring loop in 'compute_score' and then just subtracting it here, in line with original SURF* paper.
            if len(NN_far) > 0:
                scores[feature_num] -= compute_score(attr, mcmap, NN_far, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
    return scores


def MultiSURF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, headers, class_type, X, y, labels_std, data_type, weights=None):
    """ Unique scoring procedure for MultiSURF algorithm. Scoring based on 'extreme' nearest neighbors within defined radius of current target instance. """
    scores = np.zeros(num_attributes)
    if isinstance(weights, np.ndarray):
        for feature_num in range(num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += weights[feature_num] * compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
    else:
        for feature_num in range(num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)

    return scores

def MultiSURFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std, data_type, weights=None):
    """ Unique scoring procedure for MultiSURFstar algorithm. Scoring based on 'extreme' nearest neighbors within defined radius, as
    well as 'anti-scoring' of extreme far instances defined by outer radius of current target instance. """
    scores = np.zeros(num_attributes)

    if isinstance(weights, np.ndarray):
        for feature_num in range(num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += weights[feature_num] * compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
            # Note that we add this term because we used the far scoring above by setting 'near' to False.  This is in line with original MultiSURF* paper.
            if len(NN_far) > 0:
                scores[feature_num] += weights[feature_num] * compute_score(attr, mcmap, NN_far, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, near=False)
    else:
        for feature_num in range(num_attributes):
            if len(NN_near) > 0:
                scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type)
            # Note that we add this term because we used the far scoring above by setting 'near' to False.  This is in line with original MultiSURF* paper.
            if len(NN_far) > 0:
                scores[feature_num] += compute_score(attr, mcmap, NN_far, feature_num, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, near=False)

    return scores