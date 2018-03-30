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


def get_row_missing(xc, xd, cdiffs, index, cindices, dindices): #(Subset of continuous-valued feature data, Subset of discrete-valued feature data, max/min difference, instance index, boolean mask for continuous, boolean mask for discrete)
    """ Calculate distance between index instance and all other instances. """
    row = np.empty(0, dtype=np.double) #initialize empty row
    cinst1 = xc[index] #continuous-valued features for index instance
    dinst1 = xd[index] #discrete-valued features for index instance
    can = cindices[index] #Boolean mask locating missing values for continuous features for index instance
    dan = dindices[index] #Boolean mask locating missing values for discrete features for index instance
    tf = len(cinst1) + len(dinst1) #total number of features. 
    
    #Progressively compare current instance to all others. Excludes comparison with self indexed instance. (Building the distance matrix triangle).
    for j in range(index):
        dist = 0
        dinst2 = xd[j] #discrete-valued features for compared instance
        cinst2 = xc[j] #continuous-valued features for compared instance

        # Manage missing values in discrete features
        dbn = dindices[j] #Boolean mask locating missing values for discrete features for compared instance
        idx = np.unique(np.append(dan, dbn)) #indexes where there is at least one missing value in the feature between an instance pair. 
        dmc = len(idx) # Number of features excluded from distance calculation due to one or two missing values within instance pair. Used to normalize distance values for comparison.
        d1 = np.delete(dinst1, idx) #delete unique missing features from index instance
        d2 = np.delete(dinst2, idx) #delete unique missing features from compared instance
        
        # Manage missing values in continuous features
        cbn = cindices[j] #Boolean mask locating missing values for continuous features for compared instance
        idx = np.unique(np.append(can, cbn)) #indexes where there is at least one missing value in the feature between an instance pair. 
        cmc = len(idx) # Number of features excluded from distance calculation due to one or two missing values within instance pair. Used to normalize distance values for comparison.
        c1 = np.delete(cinst1, idx) #delete unique missing features from index instance
        c2 = np.delete(cinst2, idx) #delete unique missing features from compared instance
        cdf = np.delete(cdiffs, idx) #delete unique missing features from continuous value difference scores

        # Add discrete feature distance contributions (missing values excluded) - Hamming distance
        dist += len(d1[d1 != d2])

        # Add continuous feature distance contributions (missing values excluded) - Manhattan distance (Note that 0-1 continuous value normalization is included ~ subtraction of minimums cancel out)
        dist += np.sum(np.absolute(np.subtract(c1, c2)) / cdf)
        
        #Normalize distance calculation based on total number of missing values bypassed in either discrete or continuous features.
        tnmc = tf - dmc - cmc #Total number of unique missing counted
        dist = dist/float(tnmc) #Distance normalized by number of features included in distance sum (this seeks to handle missing values neutrally in distance calculation) 

        row = np.append(row, dist)
        
    return row


def compute_score(attr, mcmap, NN, feature, inst, nan_entries, headers, class_type, X, y, labels_std, near=True):
    """Flexible feature scoring method that can be used with any core Relief-based method. Scoring proceeds differently
    based on whether endpoint is binary, multiclass, or continuous. This method is called for a single target instance 
    + feature combination and runs over all items in NN. """

    fname = headers[feature] #feature identifier
    ftype = attr[fname][0]  # feature type
    ctype = class_type  # class type (binary, multiclass, continuous)
    diff_hit = diff_miss = 0.0  # Tracks the score contribution
    count_hit = count_miss = 0.0 #Tracks the number of hits/misses. Used in normalizing scores by 'k' in ReliefF, and by m or h in SURF, SURF*, MultiSURF*, and MultiSURF
    diff = 0 #Initialize 'diff' (The score contribution for this target instance and feature over all NN) 
    mmdiff = attr[fname][3] # Max/Min range of values for target feature
    
    if nan_entries[inst][feature]: #If target instance is missing, then a 'neutral' score contribution of 0 is returned immediately since all NN comparisons will be against this missing value.
        return 0.
    #Note missing data normalization below regarding missing NN feature values is accomplished by counting hits and misses (missing values are not counted) (happens in parallel with hit/miss imbalance normalization)

    xinstfeature = X[inst][feature]  #value of target instances target feature. 

    #--------------------------------------------------------------------------
    if ctype == 'binary':
        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]: #skip any NN with a missing value for this feature.
                continue

            xNNifeature = X[NN[i]][feature]
            absvalue = abs(xinstfeature - xNNifeature) / mmdiff  #Normalize absolute value of feature value difference by max-min value range for feature (so score update lies between 0 and 1)

            if near: #SCORING FOR NEAR INSTANCES
                if y[inst] == y[NN[i]]:   # HIT
                    count_hit += 1
                    if ftype == 'continuous': 
                        diff_hit -= absvalue
                    else: # discrete feature
                        if xinstfeature != xNNifeature: # A difference in feature value is observed
                            diff_hit -= 1 # Feature score is reduced when we observe feature difference between 'near' instances with the same class.
                else:  # MISS
                    count_miss += 1
                    if ftype == 'continuous':
                        diff_miss += absvalue
                    else: #discrete feature
                        if xinstfeature != xNNifeature: # A difference in feature value is observed
                            diff_miss += 1 # Feature score is increase when we observe feature difference between 'near' instances with different class values.
                            
            else:  #SCORING FOR FAR INSTANCES
                if y[inst] == y[NN[i]]:   # HIT
                    count_hit += 1
                    if ftype == 'continuous':
                        diff_hit += absvalue  #Hits differently add continuous value differences rather than subtract them 
                    else: #discrete feature
                        if xinstfeature == xNNifeature: # The same feature value is observed (Used for more efficient 'far' scoring, since there should be fewer same values for 'far' instances)
                            diff_hit -= 1 # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
                else:  # MISS
                    count_miss += 1
                    if ftype == 'continuous':
                        diff_miss -= absvalue #Misses differntly subtract continuous value differences rather than add them 
                    else: #discrete feature
                        if xinstfeature == xNNifeature: # The same feature value is observed (Used for more efficient 'far' scoring, since there should be fewer same values for 'far' instances)
                            diff_miss += 1 # Feature score is increased when we observe the same feature value between 'far' instances withh different class values.

        """ Score Normalizations:
        *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
        *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)"""
        diff = ((diff_hit / count_hit) + (diff_miss / count_miss)) / self._datalen

    #--------------------------------------------------------------------------
    elif ctype == 'multiclass':
        class_store = dict()
        missClassPSum = 0

        for each in mcmap:
            if(each != y[inst]):
                class_store[each] = [0, 0]
                missClassPSum += mcmap[each]

        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:
                continue

            xNNifeature = X[NN[i]][feature]
            absvalue = abs(xinstfeature - xNNifeature) / mmdiff
            
            if near:
                if(y[inst] == y[NN[i]]):  # HIT
                    count_hit += 1
                    if xinstfeature != xNNifeature:
                        if ftype == 'continuous':
                            diff_hit -= absvalue
                        else:
                            diff_hit -= 1
                else:  # MISS
                    for missClass in class_store:
                        if(y[NN[i]] == missClass):
                            class_store[missClass][0] += 1
                            if xinstfeature != xNNifeature:
                                if ftype == 'continuous':
                                    class_store[missClass][1] += absvalue
                                else:
                                    class_store[missClass][1] += 1
            else:  # far
                if(y[inst] == y[NN[i]]):  # HIT
                    count_hit += 1
                    if xinstfeature == xNNifeature:
                        if ftype == 'continuous':
                            diff_hit -= absvalue
                        else:
                            diff_hit -= 1
                else:  # MISS
                    for missClass in class_store:
                        if(y[NN[i]] == missClass):
                            class_store[missClass][0] += 1
                            if xinstfeature == xNNifeature:
                                if ftype == 'continuous':
                                    class_store[missClass][1] += absvalue
                                else:
                                    class_store[missClass][1] += 1

        # Corrects for both multiple classes, as well as missing data.
        missSum = 0
        for each in class_store:
            missSum += class_store[each][0]
        missAvg = missSum/float(len(class_store))

        hit_proportion = count_hit/float(len(NN))  # correct for missing data
        for each in class_store:
            diff += (mcmap[each]/float(missClassPSum)) * class_store[each][1]

        diff = diff * hit_proportion
        miss_proportion = missAvg/float(len(NN))
        diff += diff_hit * miss_proportion

        return diff

    #--------------------------------------------------------------------------
    else:  # CONTINUOUS endpoint
        same_class_bound = labels_std

        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:
                continue

            xNNifeature = X[NN[i]][feature]
            absvalue = abs(xinstfeature - xNNifeature) / mmdiff

            if near:
                if abs(y[inst] - y[NN[i]]) < same_class_bound:  # HIT
                    count_hit += 1
                    if xinstfeature != xNNifeature:
                        if ftype == 'continuous':
                            diff_hit -= absvalue
                        else:  # discrete
                            diff_hit -= 1
                else:  # MISS
                    count_miss += 1
                    if xinstfeature != xNNifeature:
                        if ftype == 'continuous':
                            diff_miss += absvalue
                        else:  # discrete
                            diff_miss += 1
            else:  # far
                if abs(y[inst] - y[NN[i]]) < same_class_bound:  # HIT
                    count_hit += 1
                    if xinstfeature == xNNifeature:
                        if ftype == 'continuous':
                            diff_hit -= absvalue
                        else:  # discrete
                            diff_hit -= 1
                else:  # MISS
                    count_miss += 1
                    if xinstfeature == xNNifeature:
                        if ftype == 'continuous':
                            diff_miss += absvalue
                        else:  # discrete
                            diff_miss += 1
                            

        hit_proportion = count_hit / float(len(NN))
        miss_proportion = count_miss / float(len(NN))
        diff = diff_hit * miss_proportion + diff_miss * hit_proportion

    return diff


def ReliefF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN, headers, class_type, X, y, labels_std):
    """ Unique scoring procedure for ReliefF algorithm. Scoring based on k nearest hits and misses of current target instance. """
    scores = np.zeros(num_attributes)
    for feature_num in range(num_attributes):
        scores[feature_num] += compute_score(attr, mcmap, NN, feature_num, inst,
                                             nan_entries, headers, class_type, X, y, labels_std)
    return scores


def SURF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN, headers, class_type, X, y, labels_std):
    """ Unique scoring procedure for SURF algorithm. Scoring based on nearest neighbors within defined radius of current target instance. """
    scores = np.zeros(num_attributes)
    if len(NN) <= 0:
        return scores
    for feature_num in range(num_attributes):
        scores[feature_num] += compute_score(attr, mcmap, NN, feature_num, inst,
                                             nan_entries, headers, class_type, X, y, labels_std)
    return scores


def SURFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std):
    """ Unique scoring procedure for SURFstar algorithm. Scoring based on nearest neighbors within defined radius, as well as 
    'anti-scoring' of far instances outside of radius of current target instance"""
    scores = np.zeros(num_attributes)
    for feature_num in range(num_attributes):
        if len(NN_near) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std)
        if len(NN_far) > 0:
            scores[feature_num] -= compute_score(attr, mcmap, NN_far, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std)
    return scores


def MultiSURF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, headers, class_type, X, y, labels_std):
    """ Unique scoring procedure for MultiSURF algorithm. Scoring based on 'extreme' nearest neighbors within defined radius of current target instance. """
    scores = np.zeros(num_attributes)
    for feature_num in range(num_attributes):
        if len(NN_near) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std)

    return scores


def MultiSURFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std):
    """ Unique scoring procedure for MultiSURFstar algorithm. Scoring based on 'extreme' nearest neighbors within defined radius, as
    well as 'anti-scoring' of extreme far instances defined by outer radius of current target instance. """
    scores = np.zeros(num_attributes)

    for feature_num in range(num_attributes):
        if len(NN_near) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std)
        if len(NN_far) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_far, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, near=False)

    return scores
