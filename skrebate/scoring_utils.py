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

import numpy as np


# (Subset of continuous-valued feature data, Subset of discrete-valued feature data, max/min difference, instance index, boolean mask for continuous, boolean mask for discrete)
def get_row_missing(xc, xd, cdiffs, index, cindices, dindices):
    """ Calculate distance between index instance and all other instances. """
    row = np.empty(0, dtype=np.double)  # initialize empty row
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

        # Add continuous feature distance contributions (missing values excluded) - Manhattan distance (Note that 0-1 continuous value normalization is included ~ subtraction of minimums cancel out)
        dist += np.sum(np.absolute(np.subtract(c1, c2)) / cdf)

        # Normalize distance calculation based on total number of missing values bypassed in either discrete or continuous features.
        tnmc = tf - dmc - cmc  # Total number of unique missing counted
        # Distance normalized by number of features included in distance sum (this seeks to handle missing values neutrally in distance calculation)
        dist = dist/float(tnmc)

        row = np.append(row, dist)

    return row


def ramp_function(data_type, attr, fname, xinstfeature, xNNifeature):
    """ Our own user simplified variation of the ramp function suggested by Hong 1994, 1997. Hong's method requires the user to specifiy two thresholds
    that indicate the max difference before a score of 1 is given, as well a min difference before a score of 0 is given, and any in the middle get a
    score that is the normalized difference between the two continuous feature values. This was done because when discrete and continuous features were mixed,
    continuous feature scores were underestimated.  Towards simplicity, automation, and a dataset adaptable approach,
    here we simply check whether the difference is greater than the standard deviation for the given feature; if so we assign a score of 1, otherwise we
    assign the normalized feature score difference.  This should help compensate for the underestimation. """
    diff = 0
    mmdiff = attr[fname][3]  # Max/Min range of values for target feature
    rawfd = abs(xinstfeature - xNNifeature)  # prenormalized feature value difference

    if data_type == 'mixed':  # Ramp function utilized
        # Check whether feature value difference is greater than the standard deviation
        standDev = attr[fname][4]
        if rawfd > standDev:  # feature value difference is is wider than a standard deviation
            diff = 1
        else:
            diff = abs(xinstfeature - xNNifeature) / mmdiff

    else:  # Normal continuous feature scoring
        diff = abs(xinstfeature - xNNifeature) / mmdiff

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

    #--------------------------------------------------------------------------
    if ctype == 'binary':
        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
                continue

            xNNifeature = X[NN[i]][feature]

            if near:  # SCORING FOR NEAR INSTANCES
                if y[inst] == y[NN[i]]:   # HIT
                    count_hit += 1
                    if ftype == 'continuous':
                        # diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff #Normalize absolute value of feature value difference by max-min value range for feature (so score update lies between 0 and 1)
                        diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)
                    else:  # discrete feature
                        if xinstfeature != xNNifeature:  # A difference in feature value is observed
                            # Feature score is reduced when we observe feature difference between 'near' instances with the same class.
                            diff_hit -= 1
                else:  # MISS
                    count_miss += 1
                    if ftype == 'continuous':
                        #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff
                        diff_miss += ramp_function(data_type, attr, fname,
                                                   xinstfeature, xNNifeature)
                    else:  # discrete feature
                        if xinstfeature != xNNifeature:  # A difference in feature value is observed
                            # Feature score is increase when we observe feature difference between 'near' instances with different class values.
                            diff_miss += 1

            else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
                if y[inst] == y[NN[i]]:   # HIT
                    count_hit += 1
                    if ftype == 'continuous':

                        #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff  #Hits differently add continuous value differences rather than subtract them 
                        diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete feature
                        if xinstfeature == xNNifeature: # The same feature value is observed (Used for more efficient 'far' scoring, since there should be fewer same values for 'far' instances)
                            diff_hit -= 1 # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
                else:  # MISS
                    count_miss += 1
                    if ftype == 'continuous':
                        #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff #Misses differntly subtract continuous value differences rather than add them 
                        diff_miss += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete feature
                        if xinstfeature == xNNifeature: # The same feature value is observed (Used for more efficient 'far' scoring, since there should be fewer same values for 'far' instances)
                            diff_miss += 1 # Feature score is increased when we observe the same feature value between 'far' instances with different class values.

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
        class_store = dict() #only 'miss' classes will be stored
        #missClassPSum = 0

        for each in mcmap:
            if(each != y[inst]):  # Identify miss classes for current target instance.
                class_store[each] = [0, 0]
                #missClassPSum += mcmap[each]

        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
                continue

            xNNifeature = X[NN[i]][feature]

            if near:  # SCORING FOR NEAR INSTANCES
                if(y[inst] == y[NN[i]]):  # HIT
                    count_hit += 1
                    if ftype == 'continuous':
                        #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff
                        diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature) 
                    else: #discrete feature
                        if xinstfeature != xNNifeature:
                            # Feature score is reduced when we observe feature difference between 'near' instances with the same class.
                            diff_hit -= 1
                else:  # MISS
                    for missClass in class_store:
                        if(y[NN[i]] == missClass):  # Identify which miss class is present
                            class_store[missClass][0] += 1
                            if ftype == 'continuous':
                                #class_store[missClass][1] += abs(xinstfeature - xNNifeature) / mmdiff
                                class_store[missClass][1] += ramp_function(
                                    data_type, attr, fname, xinstfeature, xNNifeature)
                            else:  # discrete feature
                                if xinstfeature != xNNifeature:
                                    # Feature score is increase when we observe feature difference between 'near' instances with different class values.
                                    class_store[missClass][1] += 1

            else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
                if(y[inst] == y[NN[i]]):  # HIT
                    count_hit += 1
                    if ftype == 'continuous':
                        #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff  #Hits differently add continuous value differences rather than subtract them 
                        diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete features
                        if xinstfeature == xNNifeature:
                            # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
                            diff_hit -= 1
                else:  # MISS
                    for missClass in class_store:
                        if(y[NN[i]] == missClass):
                            class_store[missClass][0] += 1
                            if ftype == 'continuous':
                                #class_store[missClass][1] += abs(xinstfeature - xNNifeature) / mmdiff
                                class_store[missClass][1] += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                            else: #discrete feature
                                if xinstfeature == xNNifeature:
                                    # Feature score is increased when we observe the same feature value between 'far' instances with different class values.
                                    class_store[missClass][1] += 1

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

        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
                continue

            xNNifeature = X[NN[i]][feature]

            if near:  # SCORING FOR NEAR INSTANCES
                if abs(y[inst] - y[NN[i]]) < same_class_bound:  # HIT approximation
                    count_hit += 1
                    if ftype == 'continuous':
                        #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff
                        diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)
                    else:  # discrete feature
                        if xinstfeature != xNNifeature:
                            # Feature score is reduced when we observe feature difference between 'near' instances with the same 'class'.
                            diff_hit -= 1
                else:  # MISS approximation
                    count_miss += 1
                    if ftype == 'continuous':
                        #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff
                        diff_miss += ramp_function(data_type, attr, fname,
                                                   xinstfeature, xNNifeature)
                    else:  # discrete feature
                        if xinstfeature != xNNifeature:
                            # Feature score is increase when we observe feature difference between 'near' instances with different class value.
                            diff_miss += 1

            else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
                if abs(y[inst] - y[NN[i]]) < same_class_bound:  # HIT approximation
                    count_hit += 1
                    if ftype == 'continuous':
                        #diff_hit += abs(xinstfeature - xNNifeature) / mmdiff
                        diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete feature
                        if xinstfeature == xNNifeature:
                            # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
                            diff_hit -= 1
                else:  # MISS approximation
                    count_miss += 1
                    if ftype == 'continuous':
                        #diff_miss -= abs(xinstfeature - xNNifeature) / mmdiff
                        diff_miss += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete feature
                        if xinstfeature == xNNifeature:
                            # Feature score is increased when we observe the same feature value between 'far' instances with different class values.
                            diff_miss += 1

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


def ReliefF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for ReliefF algorithm. Scoring based on k nearest hits and misses of current target instance. """
    scores = np.zeros(num_attributes)
    for feature_num in range(num_attributes):
        scores[feature_num] += compute_score(attr, mcmap, NN, feature_num, inst,
                                             nan_entries, headers, class_type, X, y, labels_std, data_type)
    return scores


def SURF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for SURF algorithm. Scoring based on nearest neighbors within defined radius of current target instance. """
    scores = np.zeros(num_attributes)
    if len(NN) <= 0:
        return scores
    for feature_num in range(num_attributes):
        scores[feature_num] += compute_score(attr, mcmap, NN, feature_num, inst,
                                             nan_entries, headers, class_type, X, y, labels_std, data_type)
    return scores


def SURFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for SURFstar algorithm. Scoring based on nearest neighbors within defined radius, as well as
    'anti-scoring' of far instances outside of radius of current target instance"""
    scores = np.zeros(num_attributes)
    for feature_num in range(num_attributes):
        if len(NN_near) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type)
        # Note that we are using the near scoring loop in 'compute_score' and then just subtracting it here, in line with original SURF* paper.
        if len(NN_far) > 0:
            scores[feature_num] -= compute_score(attr, mcmap, NN_far, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type)
    return scores


def MultiSURF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for MultiSURF algorithm. Scoring based on 'extreme' nearest neighbors within defined radius of current target instance. """
    scores = np.zeros(num_attributes)
    for feature_num in range(num_attributes):
        if len(NN_near) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type)

    return scores


def MultiSURFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for MultiSURFstar algorithm. Scoring based on 'extreme' nearest neighbors within defined radius, as
    well as 'anti-scoring' of extreme far instances defined by outer radius of current target instance. """
    scores = np.zeros(num_attributes)

    for feature_num in range(num_attributes):
        if len(NN_near) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type)
        # Note that we add this term because we used the far scoring above by setting 'near' to False.  This is in line with original MultiSURF* paper.
        if len(NN_far) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_far, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type, near=False)

    return scores
