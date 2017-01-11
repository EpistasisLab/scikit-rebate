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
import time as tm
from numpy import isnan, where, append, unique, delete, empty, double, array
from numpy import std, subtract, logical_not, max, min, sum, absolute
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

class SURF(BaseEstimator):

    """Feature selection using data-mined expert knowledge.

    Based on the SURF algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """
    def __init__(self, n_features_to_select=10, dlimit=10, verbose=False):
        """Sets up SURF to perform feature selection.

        Parameters
        ----------
        n_features_to_select: int (default: 10)
            the number of top features (according to the relieff score) to 
            retain after feature selection is applied.
        dlimit: int (default: 10)
            Value used to determine if a feature is discrete or continuous.
            If the number of unique levels in a feature is > dlimit, then it is
            considered continuous, or discrete otherwise.
        verbose: bool (default: False)
            if True, output timing of distance array and scoring

        """
        self.n_features_to_select = n_features_to_select
        self.dlimit = dlimit
        self.verbose = verbose
        self.headers = None
        self.feature_importances_ = None
        self.top_features_ = None

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
        Copy of the SURF instance

        """
        self.x = X
        self.y = y
        self._distance_array = None
        
        # Set up the properties for SURF
        self.datalen = len(self.x)
        self.phenotype_list = list(set(self.y))
        self.discrete_phenotype = (len(self.phenotype_list) <= self.dlimit)

        if(self.discrete_phenotype and len(self.phenotype_list) > 2):
            self.class_type = 'multiclass'
        elif(self.discrete_phenotype):
            self.class_type = 'discrete'
        else:
            self.class_type = 'continuous'
        
        self.num_attributes = len(self.x[0])

        xlen = len(self.x[0])
        mxlen = len(str(xlen+1))
        self.header = ['X' + str(i).zfill(mxlen) for i in range(1, xlen + 1)]

        # Compute the distance array between all data points
        start = tm.time()
        if(self.mdcnt > 0 or self.data_type == 'mixed'):
            attr = self.get_attribute_info()
            diffs,cidx,didx = self.dtypeArray(attr)
            cdiffs = diffs[cidx]
            xc = self.x[:,cidx]
            xd = self.x[:,didx]
            self._distance_array = self.distarray_mixed_missing(xc, xd, cdiffs)
        else:
            self._distance_array = self.distarray_clean()
            
        if self.verbose:
            elapsed = tm.time() - start
            print('Created distance array in ' + str(elapsed) + ' seconds.')
            print('SURF scoring under way ...')
            
        start = tm.time()
        self.feature_importances_ = np.array(self.runSURF())

        if self.verbose:
            elapsed = tm.time() - start
            print('Completed scoring in ' + str(elapsed) + ' seconds.')

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

############################# Properties ###############################
    #==================================================================#    
    @property 
    def mdcnt(self):  
        """ missing data count """
        return isnan(self.x).sum()
    #==================================================================#    
    @property
    def phenSD(self):
        """ standard deviation of class if continuous """
        if(len(self.phenotype_list) <= self.dlimit):
            return 0
        else:
            return std(self.y, ddof=1)
    #==================================================================#
    def get_attribute_info(self):
        attr = dict()
        d = 0
        limit = self.dlimit
        w = self.x.transpose()
        md = self.mdcnt
        
        for idx in range(len(w)):
            h = self.header[idx]
            z = w[idx]
            if(md > 0): z = z[logical_not(isnan(z))]
            zlen = len(unique(z)) 
            if(zlen <= limit):
                attr[h] = ('discrete',0,0,0)
                d += 1
            else:
                mx = max(z)
                mn = min(z)
                attr[h] = ('continuous',mx, mn, mx - mn)
        
        return attr
    #==================================================================#    
    @property
    def data_type(self):
        C = D = False
        
        attr = self.get_attribute_info()
        
        for key in attr.keys():
            if(attr[key][0] == 'discrete'): D = True
            if(attr[key][0] == 'continuous'): C = True
                
        if(C and D): 
            return 'mixed'
        elif(D and not C):
            return 'discrete'
        elif(C and not D):
            return 'continuous'
    #==================================================================#    
    def distarray_clean(self):
        """ distance array for clean contiguous data """
        from scipy.spatial.distance import pdist, squareform
        attr = self.get_attribute_info()
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
        if(self.data_type == 'discrete'):
            return squareform(pdist(self.x, metric='hamming'))
        else:
            self.x = pre_normalize(self.x)
            return squareform(pdist(self.x, metric='cityblock'))
    
######################### SUPPORTING METHODS ###########################
    def dtypeArray(self,attr):
        """  Return mask for discrete(0)/continuous(1) attributes and their 
             indices. Return array of max/min diffs of attributes. """
        attrtype = []
        attrdiff = []
        
        for key in self.header:
            if(attr[key][0] == 'continuous'):
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
    def distarray_mixed_missing(self, xc, xd, cdiffs):
        """ distance array for mixed/missing data """
        
        dist_array = []
        missing = self.mdcnt
        
        if(missing > 0):
            cindices = []
            dindices = []
            for i in range(self.datalen):
                cindices.append(where(isnan(xc[i]))[0])
                dindices.append(where(isnan(xd[i]))[0])
        
        for index in range(self.datalen):
            if(missing > 0):
                row = self.get_row_missing(xc, xd, cdiffs, index, 
                                           cindices, dindices)
            else:
                row = self.get_row_mixed(xc, xd, cdiffs, index)
                
            row = list(row)
            dist_array.append(row)
            
        return dist_array
    #==================================================================#    
    def get_row_missing(self, xc, xd, cdiffs, index, cindices, dindices):

        row = empty(0,dtype=double)
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
            idx = unique(append(can,cbn))   # create unique list
            c1 = delete(cinst1,idx)       # remove elements by idx
            c2 = delete(cinst2,idx)
            cdf = delete(cdiffs,idx)

            # discrete
            dbn = dindices[j]
            idx = unique(append(dan,dbn))
            d1 = delete(dinst1,idx)
            d2 = delete(dinst2,idx)
            
            # discrete first
            dist += len(d1[d1 != d2])

            # now continuous
            dist += sum(absolute(subtract(c1,c2)) / cdf)

            row = append(row,dist)

        return row
    #==================================================================#    
    def get_row_mixed(self, xc, xd, cdiffs, index):

        row = empty(0,dtype=double)
        d1 = xd[index]
        c1 = xc[index]
        for j in range(index):
            dist = 0
            d2 = xd[j]
            c2 = xc[j]
    
            # discrete first
            dist += len(d1[d1 != d2])

            # now continuous
            dist += sum(absolute(subtract(c1,c2)) / cdiffs)
    
            row = append(row,dist)

        return row
    
############################# SURF ############################################
    def find_nearest_neighbor(self, inst, avgDist):  # for SURF
        NN = []
        min_indicies = []

        for i in range(self.datalen):
            if(inst != i):
                locator = [inst,i]
                if(i > inst): locator.reverse()
                d = self._distance_array[locator[0]][locator[1]]
                if(d < avgDist):
                    min_indicies.append(i)

        for i in range(len(min_indicies)):
            NN.append(min_indicies[i])

        return NN

    def compute_scores(self, inst, attr, mcmap, nan_entries, avgDist):
        scores = np.zeros(self.num_attributes)
        NN = self.find_nearest_neighbor(inst, avgDist)
        NN = np.array(NN, dtype=np.int32)
        if(len(NN) <= 0): return
        for feature_num in range(self.num_attributes):
            scores[feature_num] += self.evaluate_SURF(attr, NN, feature_num, inst, mcmap, nan_entries)
        return scores

    def runSURF(self):
        #---------------------------------------------------------------------
        # Find number of classes in the dataset and store them into the map
        def getMultiClassMap():
            mcmap = dict()
            y = self.y
    
            for i in range(self.datalen):
                if(self.y[i] not in mcmap):
                    mcmap[self.y[i]] = 0
                else:
                    mcmap[self.y[i]] += 1
    
            for each in self.phenotype_list:
                mcmap[each] = mcmap[each]/float(maxInst)

            return mcmap
    
        #------------------------------#
        # calculate avgDist
        sm = cnt = 0
        for i in range(self.datalen):
            sm += sum(self._distance_array[i])
            cnt += len(self._distance_array[i])
        avgDist = sm/float(cnt)
        #------------------------------#
    
        if(self.class_type == 'multiclass'):
            mcmap = getMultiClassMap()
        else:
            mcmap = 0
        
        attr = self.get_attribute_info()
        scores = np.sum(Parallel(n_jobs=-1)(delayed(self.compute_scores)(instance_num, attr, mcmap, isnan(self.x), avgDist) for instance_num in range(self.datalen)), axis=0)
    
        return scores

    ###############################################################################
    def evaluate_SURF(self, attr, NN, feature, inst, mcmap, nan_entries):
        """ evaluates both SURF and SURF* scores """
    
        fname = self.header[feature]
        ftype = attr[fname][0]  # feature type
        ctype = self.class_type # class type
        diff_hit = diff_miss = 0.0 
        count_hit = count_miss = 0.0
        mmdiff = 1
        diff = 0

        if nan_entries[inst][feature]:
            return 0.

        xinstfeature = self.x[inst][feature]
    
        if ctype == 'multiclass':
            class_store = dict()
            miss_class_psum = 0   # for SURF
            for each in mcmap:
                if each != self.y[inst]:
                    class_store[each] = [0,0]
                    miss_class_psum += mcmap[each]  # for SURF
    
            for i in range(len(NN)):
                if nan_entries[NN[i]][feature]:
                    continue

                NN[i] = int(NN[i])
                xNNifeature = self.x[NN[i]][feature]
                absvalue = abs(xinstfeature - xNNifeature) / mmdiff
    
                if self.y[inst] == self.y[NN[i]]:  # HIT
                    count_hit += 1
                    if xinstfeature != xNNifeature:
                        if ftype == 'continuous':
                            diff_hit -= absvalue
                        else:  # discrete
                            diff_hit -= 1
    
                else:  # MISS
                    for miss_class in class_store:
                        if self.y[NN[i]] == miss_class:
                            class_store[miss_class][0] += 1
                            if xinstfeature != xNNifeature:
                                if ftype == 'continuous':
                                    class_store[miss_class][1] += absvalue
                                else:  # discrete
                                    class_store[miss_class][1] += 1
    
            # corrects for both multiple classes as well as missing data
            missSum = 0
            for each in class_store:
                missSum += class_store[each][0]
            missAverage = missSum/float(len(class_store))
    
            hit_proportion = count_hit / float(len(NN)) # Correct for NA
            for each in class_store:
                diff_miss += (mcmap[each] / float(miss_class_psum)) * class_store[each][1]
    
            diff = diff_miss * hit_proportion
            miss_proportion = missAverage / float(len(NN))
            diff += diff_hit * miss_proportion
    
        #--------------------------------------------------------------------------
        elif ctype == 'discrete':
            for i in range(len(NN)):
                if nan_entries[NN[i]][feature]:
                    continue

                xNNifeature = self.x[NN[i]][feature]
                absvalue = abs(xinstfeature - xNNifeature) / mmdiff
    
                if self.y[inst] == self.y[NN[i]]:   # HIT
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
            same_class_bound = self.phenSD

            for i in range(len(NN)):
                if nan_entries[NN[i]][feature]:
                    continue

                xNNifeature = self.x[NN[i]][feature]
                absvalue = abs(xinstfeature - xNNifeature) / mmdiff
    
                if abs(self.y[inst] - self.y[NN[i]]) < same_class_bound: # HIT
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

def main():
    import numpy as np
    import pandas as pd

    data = pd.read_csv('~/Downloads/VDR-data/VDR_Data.tsv', sep='\t').sample(frac=1.)
    features = data.drop('class', axis=1).values
    labels = data['class'].values

    clf = SURF()
    clf.fit(features, labels)

    print(data.columns[np.argsort(clf.feature_importances_)][::-1])
    print(clf.feature_importances_[np.argsort(clf.feature_importances_)][::-1])

if __name__ == '__main__':
    main()
