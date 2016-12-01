# -*- coding: utf-8 -*-

"""
Mon Sep 19 09:55:01 EDT 2016
"""

from __future__ import print_function
import numpy as np
import sys
import math
import time as tm
from numpy import isnan, where, append, unique, delete, empty, double, array
from numpy import std, subtract, logical_not, max, min, sum, absolute
from numpy import subtract

class ReliefF(object):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """
    def __init__(self, pname='Class', missing='NA', verbose=False, 
                       n_neighbors=10, dlimit=10, n_features_to_keep=10,
                       hdr=None):
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        pname: str (default: 'Class') 
            name of phenotype 
        missing: str (default: 'NA') 
            missing data value
        verbose: bool (default = False)
            If True output creation times of both distance array and scores
        n_neighbors: int (default: 10)
            The number of neighbors to consider when assigning feature
            importance scores.
            More neighbors results in more accurate scores, but takes longer.
        dlimit: int (default: 10)
            max value to determine if attributes/class are discrete
        n_features_to_keep: int (default: 10)
            The number of top features (according to the ReliefF score) to 
            retain after feature selection is applied.
        hdr: list (default: None)
            This is a list of attribute names for the data.  If this is  None
            one will be created by the header property.

        Returns
        -------
        None

        """
        self.phenotype_name = pname
        self.dlimit = dlimit  # discrete limit
        self.missing = missing
        self.verbose = verbose
        self.hdr = hdr
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep
        self.feature_scores = None
        self.top_features = None
        self.Scores = None

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
        None
        """    
        self.x = X
        self.y = y
        
        start = tm.time()
        self.Scores = self.runRelieff()
        self.feature_scores = np.array(self.Scores)
        # Compute indices of top features, cast scores to floating point.
        self.top_features = np.argsort(self.feature_scores)[::-1]
        self.feature_scores = self.feature_scores.astype(np.float64)
        if(self.verbose):
            elapsed = tm.time() - start
            print('Created feature scores in ' + str(elapsed) + ' seconds')
        return self

    #=========================================================================#
    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        return X[:, self.top_features[:self.n_features_to_keep]]

    #=========================================================================#
    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X)

############################# Properties ###############################
    @property
    def header(self):
        if(self.hdr == None):
            xlen = len(self.x[0])
            mxlen = len(str(xlen+1))
            header = ['X' + str(i).zfill(mxlen) for i in range(1, xlen + 1)]
        else:
            header = self.hdr
        return header
    #==================================================================#    
    @property
    def datalen(self):
        return len(self.x)
    #==================================================================#    
    @property
    def num_attributes(self):
        return len(self.x[0])
    #==================================================================#    
    @property
    def phenotype_list(self):
        return list(set(self.y))
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
    @property
    def discrete_phenotype(self):
        if(len(self.phenotype_list) <= self.dlimit):
            return True
        else:
            return False
    #==================================================================#    
    @property
    def get_attribute_info(self):
        attr = dict()
        c = d = 0
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
        
        attr = self.get_attribute_info
        
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
    @property
    def class_type(self):
        dp = self.discrete_phenotype
        if(dp and len(self.phenotype_list) > 2):
            return 'multiclass'
        elif(dp):
            return 'discrete'
        else:
            return 'continuous'
    #==================================================================#    
    @property
    def distarray_clean(self):
        """ distance array for clean contiguous data """
        from scipy.spatial.distance import pdist, squareform
        attr = self.get_attribute_info
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
        pname = self.phenotype_name
        #attr = self.get_attribute_info
        
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
        datalen = self.datalen
        missing = self.mdcnt
        
        if(missing > 0):
            cindices = []
            dindices = []
            for i in range(datalen):
                cindices.append(where(isnan(xc[i]))[0])
                dindices.append(where(isnan(xd[i]))[0])
        
        for index in range(datalen):
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
######################### RELIEFF #############################################
    def runRelieff(self):
        """ Create scores using ReleifF from data """
        #==================================================================#
        def buildIndexLists():
            """ This creates lists of indexes of observations that share 
                the same value in the phenotype"""
            index = 0
            indicies = dict()

            # initialize dictionary to hold as many lists
            # as there are unique values in the phenotype
            for i in self.phenotype_list:
                indicies[i] = []

            for i in y:
                indicies[i].append(index)
                index += 1

            return indicies
        #==================================================================#
        def getdistance(i):
            if(i == inst):
                return sys.maxsize
            elif(i < inst):
                return distArray[inst][i]
            else:
                return distArray[i][inst]
        #==================================================================#
        def getsortTuple(x):
            return (getdistance(x),x)
        #==================================================================#

        mdcnt = self.mdcnt
        datatype = self.data_type
        attr = self.get_attribute_info
        x = self.x
        y = self.y

        start = tm.time()
        if(mdcnt > 0 or datatype == 'mixed'):
            diffs,cidx,didx = self.dtypeArray(attr)
            cdiffs = diffs[cidx]
            xc = self.x[:,cidx]
            xd = self.x[:,didx]
            distArray = self.distarray_mixed_missing(xc, xd, cdiffs)
        else:
            distArray = self.distarray_clean
        if(self.verbose):
            elapsed = tm.time() - start
            print('Created distance array in ' + str(elapsed) + ' seconds')
            print('ReliefF scoring under way ...')

        start = tm.time()
        neighbors = self.n_neighbors
        numattr = self.num_attributes
        datalen = self.datalen
        header = self.header
        maxInst = datalen
        
        Scores = [0] * numattr

        indicies = buildIndexLists()
    
        if(self.class_type == 'multiclass'):
            mcmap = self.getMultiClassMap()
        else:
            mcmap = 0

        for inst in range(datalen):
            NN = []
            for sample in indicies.values():
                n = sorted(sample,key=getsortTuple)
                NN.extend(n[:neighbors])
        
            for ai in range(numattr):
                nn = array(NN)
                Scores[ai] += self.getScores(header,attr,x,y,nn,inst,ai,mcmap)

        #averaging the scores
        divisor = maxInst * neighbors
        for ai in range(numattr):
            Scores[ai] = Scores[ai]/float(divisor)
            
        return Scores
    
    #=====================================================================#
    def getMultiClassMap(self):
        """ Find number of classes in the dataset and 
            store them into the map """
        mcmap = dict()
        y = self.y
        maxInst = self.datalen
        
        for i in range(maxInst):
            if(y[i] not in mcmap):
                mcmap[y[i]] = 0
            else:
                mcmap[y[i]] += 1

        for each in self.phenotype_list:
            mcmap[each] = mcmap[each]/float(maxInst)
        
        return mcmap
    
    #=====================================================================#
    def getScores(self, header, attr, x, y, NN, inst, ai, mcmap):
        """ Method evaluating the score of an attribute
            called from runReliefF() """
        calc = 0
        same_class_bound = self.phenSD
        lenNN = len(NN)

        hit_count = miss_count = hit_diff = miss_diff = diff = 0
        classtype = self.class_type
        datatype  = attr[header[ai]][0]
        mmdiff    = attr[header[ai]][3]
        inst_item = x[inst][ai]

        #---------------------------------------------------------------------#
        if(classtype == 'discrete'):
            for i in range(lenNN):
                NN_item = x[NN[i]][ai]
                if(math.isnan(inst_item) or math.isnan(NN_item)): continue
                if(datatype == 'continuous'):
                    calc = abs(inst_item - NN_item)/mmdiff

                if(y[inst] == y[NN[i]]):
                    hit_count += 1  # HIT
                    if(inst_item != NN_item):
                        if(datatype == 'continuous'):
                            hit_diff -= calc
                        else:
                            hit_diff -= 1
                else:  #MISS
                    miss_count += 1
                    if(inst_item != NN_item):
                        if(datatype == 'continuous'):
                            miss_diff += calc
                        else:
                            miss_diff += 1

            return hit_diff + miss_diff
        #---------------------------------------------------------------------#
        if(classtype == 'multiclass'):
            class_store = dict()
            missClassPSum = 0

            for each in mcmap:
                if(each != y[inst]):
                    class_store[each] = [0,0]
                    missClassPSum += mcmap[each]

            for i in range(len(NN)):
                NN_item = x[NN[i]][ai]
                if(math.isnan(inst_item) or math.isnan(NN_item)): continue
                if(datatype == 'continuous'):
                    calc = abs(inst_item - NN_item)/mmdiff

                if(y[inst] == y[NN[i]]): #HIT
                    hit_count += 1
                    if(inst_item != NN_item):
                        if(datatype == 'continuous'):
                            hit_diff -= calc
                        else:
                            hit_diff -= 1
                else:  #MISS
                    for missClass in class_store:
                        if(y[NN[i]] == missClass):
                            class_store[missClass][0] += 1
                            if(inst_item != NN_item):
                                if(datatype == 'continuous'):
                                    class_store[missClass][1] += calc
                                else:
                                    class_store[missClass][1] += 1

            #Corrects for both multiple classes, as well as missing data.
            missSum = 0
            for each in class_store:
                missSum += class_store[each][0]
            missAvg = missSum/float(len(class_store))

            hit_prop = hit_count/float(len(NN))  # correct for missing data
            for each in class_store:
                diff += (mcmap[each]/float(missClassPSum)) * class_store[each][1]

            diff = diff * hit_prop
            miss_prop = missAvg/float(len(NN))
            diff += hit_diff * miss_prop

            return diff
        #---------------------------------------------------------------------#
        if(classtype == 'continuous'):

            for i in range(len(NN)):
                NN_item = x[NN[i]][ai]
                if(math.isnan(inst_item) or math.isnan(NN_item)): continue
                if(datatype == 'continuous'):
                    calc = abs(inst_item - NN_item)/mmdiff

                if(abs(y[inst] - y[NN[i]]) < same_class_bound):  #HIT
                    hit_count += 1
                    if(inst_item != NN_item):
                        if(datatype == 'continuous'):
                            hit_diff -= calc
                        else:
                            hit_diff -= 1
                    else:   #MISS
                        miss_count += 1
                    if(inst_item != NN_item):
                        if(datatype == 'continuous'):
                            miss_diff += calc
                        else:
                            miss_diff += 1

            # Take hit/miss inbalance into account (coming from missing data, 
            # or inability to find enough continuous neighbors)

            hit_prop = hit_count/float(len(NN))
            miss_prop = miss_count/float(len(NN))

            return hit_diff * miss_prop + miss_diff * hit_prop
###############################################################################
