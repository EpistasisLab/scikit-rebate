# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import time as tm
from numpy import isnan, where, append, unique, delete, empty, double, array
from numpy import std, subtract, logical_not, max, min, sum, absolute, subtract

class SURFstar(object):

    """Feature selection using data-mined expert knowledge.

    Based on the SURF* algorithm as introduced in:

    moore, jason et al. multiple threshold spatially uniform relieff for
    the genetic analysis of complex human diseases. 

    """
    def __init__(self, verbose=False,
                       dlimit=10,  n_features_to_keep=10, hdr=None):
        """Sets up SURFstar to perform feature selection.

        Parameters
        ----------
        verbose: bool (default: False)
            if True, output timing of distance array and scoring
        dlimit: int (default: 10)
            max value that determines if feature/class is discrete
        n_features_to_keep: int (default: 10)
            the number of top features (according to the feature scores) to 
            retain after feature selection is applied.
        hdr: list (default: None)
            Allow user to provide header from CLI

        """
        self.dlimit = dlimit
        self.verbose = verbose
        self.n_features_to_keep = n_features_to_keep
        self.feature_scores = None
        self.top_features = None
        self.hdr = hdr

    #=========================================================================#
    def fit(self, X, y):
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        None
        
        Returns
        -------
        Copy of the SURFstar instance

        """
        self.x = X
        self.y = y
        self.Scores = None
        self.distArray = None
        #=====================================================================#
        # get distance array
        start = tm.time()
        if(self.mdcnt > 0 or self.data_type == 'mixed'):
            attr = self.get_attribute_info
            diffs,cidx,didx = self.dtypeArray(attr)
            cdiffs = diffs[cidx]
            xc = self.x[:,cidx]
            xd = self.x[:,didx]
            self.distArray = self.distarray_mixed_missing(xc, xd, cdiffs)
        else:
            self.distArray = self.distarray_clean
            
        if(self.verbose):
            elapsed = tm.time() - start
            print('Created distance array in ' + str(elapsed) + ' seconds.')
            print('SURF* scoring under way ...')
            
        start = tm.time()
        self.Scores = self.runSURFstar()

        if(self.verbose):
            elapsed = tm.time() - start
            print('Completed scoring ' + str(elapsed) + ' seconds.')
            
        self.feature_scores = np.array(self.Scores)

        # Compute indices of top features, cast scores to floating point.
        self.top_features = np.argsort(self.feature_scores)[::-1]
        self.feature_scores = self.feature_scores.astype(np.float64)
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
    
############################# SURFstar ########################################
    def runSURFstar(self):
        
        from math import isnan
        maxInst = self.datalen
        numattr = self.num_attributes
        Scores = [0] * numattr
        x = self.x
        y = self.y
        header = self.header
        attr = self.get_attribute_info
        distArray = self.distArray
        #---------------------------------------------------------------------
        def find_data_instances():  # for SURFStar
            NN_near=[]; NN_far=[]
            min_indices=[]; max_indices=[]

            for i in range(maxInst):
                if(inst != i):
                    locator = [inst,i]
                    if(i > inst): locator.reverse()
                    d = distArray[locator[0]][locator[1]]

                    if(d < avgDist): min_indices.append(i)
                    if(d > avgDist): max_indices.append(i)

            for i in range(len(min_indices)):
                NN_near.append(min_indices[i])

            for i in range(len(max_indices)):
                NN_far.append(max_indices[i])

            return NN_near, NN_far
        #---------------------------------------------------------------------
        # Find number of classes in the dataset and store them into the map
        def getMultiClassMap():
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
    
        #------------------------------#
        # calculate avgDist
        sm = cnt = 0
        for i in range(maxInst):
            sm += sum(distArray[i])
            cnt += len(distArray[i])
        avgDist = sm/float(cnt)
        #------------------------------#
    
        if(self.class_type == 'multiclass'):
            mcmap = getMultiClassMap()
        else:
            mcmap = 0
    
        for inst in range(maxInst):
            NN_near,NN_far = find_data_instances()
            NN_near = np.array(NN_near, dtype=np.int32)
            NN_far = np.array(NN_far, dtype=np.int32)
            for f in range(self.num_attributes):
                if(len(NN_near) > 0):
                    Scores[f] += \
                           self.evaluate_SURF(x,y,header,attr,NN_near,f,inst,mcmap)
                if(len(NN_far) > 0):
                    Scores[f] -= \
                           self.evaluate_SURF(x,y,header,attr,NN_far,f,inst,mcmap)
    
        return Scores
    ###############################################################################
    def evaluate_SURF(self, x, y, header, attr, NN, feature, inst, mcmap):
        """ evaluates SURF* scores """
    
        fname = header[feature]
        ftype = attr[fname][0]  # feature type
        ctype = self.class_type # class type
        diff_hit = diff_miss = 0.0 
        count_hit = count_miss = 0.0
        mmdiff = 1
        diff = 0
    
        xinstfeature = x[inst][feature]
    
        if(ftype == 'continuous'): mmdiff = attr[fname][3]
    
        if(ctype == 'multiclass'):
            class_Store = dict()
            missClassPSum = 0   # for SURF
            for each in mcmap:
                if(each != y[inst]):
                    class_Store[each] = [0,0]
                    missClassPSum += mcmap[each]  # for SURF
    
            for i in range(len(NN)):
                NN[i] = int(NN[i])
                xNNifeature = x[NN[i]][feature]
                absvalue = abs(xinstfeature - xNNifeature)/mmdiff
    
                if(isnan(xinstfeature) or isnan(xNNifeature)): continue
    
                if(y[inst] == y[NN[i]]):  # HIT
                    count_hit += 1
                    if(xinstfeature != xNNifeature):
                        if(ftype == 'continuous'):
                            diff_hit -= absvalue
                        else:  # discrete
                            diff_hit -= 1
    
                else:  # MISS
                    for missClass in class_Store:
                        if(y[NN[i]] == missClass):
                            class_Store[missClass][0] += 1
                            if(xinstfeature != xNNifeature):
                                if(ftype == 'continuous'):
                                    class_Store[missClass][1] += absvalue
                                else:  # discrete
                                    class_Store[missClass][1] += 1
    
            # corrects for both multiple classes as well as missing data
            missSum = 0
            for each in class_Store: missSum += class_Store[each][0]
            missAverage = missSum/float(len(class_Store))
    
            hit_proportion = count_hit / float(len(NN)) # Correct for NA
            for each in class_Store:
                diff_miss += (class_Store[each][0]/float(missSum)) * \
                    class_Store[each][1]
    
            diff = diff_miss * hit_proportion
            miss_proportion = missAverage / float(len(NN))
            diff += diff_hit * miss_proportion
    
        #--------------------------------------------------------------------------
        elif(ctype == 'discrete'):
            for i in range(len(NN)):
                xNNifeature = x[NN[i]][feature]
                xinstfeature = x[inst][feature]
                absvalue = abs(xinstfeature - xNNifeature)/mmdiff
    
                if(isnan(xinstfeature) or isnan(xNNifeature)): continue
    
                if(y[inst] == y[NN[i]]):   # HIT
                    count_hit += 1
                    if(xinstfeature != xNNifeature):
                        if(ftype == 'continuous'):
                            diff_hit -= absvalue
                        else: # discrete
                            diff_hit -= 1
                else: # MISS
                    count_miss += 1
                    if(xinstfeature != xNNifeature):
                        if(ftype == 'continuous'):
                            diff_miss += absvalue
                        else: # discrete
                            diff_miss += 1
    
            hit_proportion = count_hit/float(len(NN))
            miss_proportion = count_miss/float(len(NN))
            diff = diff_hit * miss_proportion + diff_miss * hit_proportion
        #--------------------------------------------------------------------------
        else: # CONTINUOUS endpoint
            same_class_bound = self.phenSD

            for i in range(len(NN)):
                xNNifeature = x[NN[i]][feature]
                xinstfeature = x[inst][feature]
                absvalue = abs(xinstfeature - xNNifeature)/mmdiff
    
                if(isnan(xinstfeature) or isnan(xNNifeature)): continue
    
                if(abs(y[inst] - y[NN[i]]) < same_class_bound): # HIT
                    count_hit += 1
                    if(xinstfeature != xNNifeature):
                        if(ftype == 'continuous'):
                            diff_hit -= absvalue
                        else: # discrete
                            diff_hit -= 1
                else: # MISS
                    count_miss += 1
                    if(xinstfeature != xNNifeature):
                        if(ftype == 'continuous'):
                            diff_miss += absvalue
                        else: # discrete
                            diff_miss += 1
    
            hit_proportion = count_hit/float(len(NN))
            miss_proportion = count_miss/float(len(NN))
            diff = diff_hit * miss_proportion + diff_miss * hit_proportion
    
        return diff
