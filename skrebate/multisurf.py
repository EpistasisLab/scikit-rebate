# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import math
import time as tm
from numpy import isnan, where, append, unique, delete, empty, double, array
from numpy import std, subtract, logical_not, max, min, sum, absolute, subtract

class MultiSURF(object):

    """Feature selection using data-mined expert knowledge.

    Based on the MultiSURF algorithm as introduced in:

    moore, jason et al. multiple threshold spatially uniform relieff for
    the genetic analysis of complex human diseases. 

    """
    def __init__(self, verbose=False,
                       dlimit=10,  n_features_to_keep=10, hdr=None):
        """Sets up MultiSURF to perform feature selection.

        Parameters
        ----------
        verbose: bool (default: False)
            if True, output timing of distance array and scoring
        dlimit: int (default: 10)
            max value that determines if feature/class is discrete
        n_features_to_keep: int (default: 10)
            the number of top features (according to the relieff score) to 
            retain after feature selection is applied.
        hdr: list (default: None)
            User can provided custom header list from CLI

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
        Copy of the MultiSURF instance

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
            print('MultiSURF scoring under way ...')
            
        start = tm.time()
        if(self.class_type == 'multiclass'):
            self.Scores = self.mcMultiSURF()
        else:
            self.Scores = self.runMultiSURF()

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
    
######################## MULTISURF ############################################
    def runMultiSURF(self):
        """ get multiSURF scores """
        distArray = self.distArray
        #----------------------------------------------------------------------
        def get_individual_distances(x):
            d=[]
            for j in range(len(x)):
                if (i!=j):
                    locator = [i,j]
                    if(i < j): locator.reverse()
                    d.append(distArray[locator[0]][locator[1]])
            return d
        #----------------------------------------------------------------------
        x = self.x
        y = self.y
        header = self.header
        attr = self.get_attribute_info
        numattr = self.num_attributes
        datalen = self.datalen
        same_class_bound = self.phenSD
        classtype = self.class_type
        mmdiff = j = 0
        ScoreList = [0] * numattr
        avg_dist = []; D = []

        for i in range(datalen):
            dist_vect = get_individual_distances(self.x)
            avg_dist.append(np.average(dist_vect))
            D.append(std(dist_vect)/2.0)

        for k in range(numattr):
            datatype = attr[header[k]][0]
            if(datatype == 'continuous'):
                mmdiff = attr[header[k]][3]

            count_hit_near = count_miss_near = 0.0
            diff_hit_near = diff_miss_near  = 0.0
            count_hit_far = count_miss_far = 0.0
            diff_hit_far = diff_miss_far   = 0.0

            for i in range(datalen):
                xik = x[i][k]
                if(isnan(xik)): continue
                for j in range(i,datalen):
                    if(i == j) : continue 
                    xjk = x[j][k]
                    if(isnan(xjk)): continue
    
                    if(datatype == 'continuous'):
                        calc = abs(xik - xjk) / mmdiff
    
                    locator = [i,j]
                    if(i < j): locator.reverse()
                    d = distArray[locator[0]][locator[1]]
    
                    #---------------------------------------------------------
                    if(d < avg_dist[i] - D[i]):  # NEAR
                     
                        if(classtype == 'discrete'):
                            if(y[i] == y[j]): # SAME ENDPOINT
                                count_hit_near += 1
                                if(xik != xjk):
                                    if(datatype == 'continuous'):
                                        diff_hit_near -= calc
                                    else:
                                        diff_hit_near -= 1
                            else: # DIFFERENT ENDPOINT
                                count_miss_near += 1
                                if(xik != xjk):
                                    if(datatype == 'continuous'):
                                        diff_miss_near += calc
                                    else:
                                        diff_miss_near += 1
    
                        else: # CONTINUOUS ENDPOINT
                            if(abs(y[i] - y[j]) < same_class_bound):
                                count_hit_near += 1
                                if(xik != xjk):
                                    if(datatype == 'continuous'):
                                        diff_hit_near -= calc
                                    else: # DISCRETE
                                        diff_hit_near -= 1
                            else:
                                count_miss_near += 1
                                if(xik != xjk):
                                    if(datatype == 'continuous'):
                                        diff_miss_near += calc
                                    else: # DISCRETE
                                        diff_miss_near += 1
                    #----------------------------------------------------------
                    if(d > avg_dist[i] + D[i]):  # FAR
                     
                        if(classtype == 'discrete'):
                            if(y[i] == y[j]):
                                count_hit_far += 1
                                if(datatype == 'continuous'):
                                    diff_hit_far -= calc
                                else: # DISCRETE
                                    if(xik == xjk): diff_hit_far -= 1
                            else:
                                count_miss_far += 1
                                if(datatype == 'continuous'):
                                    diff_miss_far += calc
                                else: # DISCRETE
                                    if(xik == xjk): diff_miss_far += 1
    
                        else: # CONTINUOUS ENDPOINT
                            if(abs(y[i] - y[j]) < same_class_bound):
                                count_hit_far += 1
                                if(datatype == 'continuous'):
                                    diff_hit_far -= calc
                                else: # DISCRETE
                                    if(xik == xjk): diff_hit_far -= 1
    
                            else:
                                count_miss_far += 1
                                if(datatype == 'continuous'):
                                    diff_miss_far += calc
                                else: #DISCRETE
                                    if(xik == xjk): diff_miss_far += 1
                    #----------------------------------------------------------
    
            hit_proportion=count_hit_near/(count_hit_near + count_miss_near)
            miss_proportion=count_miss_near/(count_hit_near + count_miss_near)
    
            #applying weighting scheme to balance the scores
            diff = diff_hit_near * miss_proportion + diff_miss_near * hit_proportion
    
            hit_proportion = count_hit_far/(count_hit_far + count_miss_far)
            miss_proportion = count_miss_far/(count_hit_far + count_miss_far)
    
            #applying weighting scheme to balance the scores
            diff += diff_hit_far * miss_proportion + diff_miss_far * hit_proportion
            
            ScoreList[k]+=diff
    
        return ScoreList

####################### MULTISURF (multiclass) ################################
    def mcMultiSURF(self):
        """ get multiSURF scores for multiclass Class
            Controls major MultiSURF loops. """
        distArray = self.distArray
        #--------------------------------------------------------------------------
        def get_individual_distances():
            d=[]
            for j in range(datalen):
                if (i!=j):
                    locator = [i,j]
                    if(i < j): locator.reverse()
                    d.append(distArray[locator[0]][locator[1]])
            return d
        #--------------------------------------------------------------------------
        def makeClassPairMap():
            """ finding number of classes in the dataset 
                and storing them into the map """
            classPair_map = dict()
            for each in multiclass_map:
                for other in multiclass_map:
                    if(each != other):
                        locator = [each,other]
                        if(each < other): locator.reverse()
                        tempString = str(locator[0]) + str(locator[1])
                        if (not classPair_map.has_key(tempString)):
                            classPair_map[tempString] = [0,0]
            return classPair_map
        #--------------------------------------------------------------------------
        def getMultiClassMap():
            """ Find number of classes in the dataset and 
                store them into the map """
            mcmap = dict()
            maxInst = self.datalen
            y = self.y
    
            for i in range(maxInst):
                if(y[i] not in mcmap):
                    mcmap[y[i]] = 0
                else:
                    mcmap[y[i]] += 1
    
            for each in self.phenotype_list:
                mcmap[each] = mcmap[each]/float(maxInst)
                
            return mcmap
        #--------------------------------------------------------------------------
        x = self.x
        y = self.y
        attr = self.get_attribute_info
        numattr = self.num_attributes
        datalen = self.datalen
        header = self.header
        mmdiff = 0.0 
        calc = calc1 = 0
    
        ScoreList=[0] * numattr
        D=[]; avg_dist=[]
        
        multiclass_map = getMultiClassMap()
    
        for i in range(datalen):
            dist_vect = get_individual_distances()
            avg_dist.append(np.average(dist_vect))
            D.append(np.std(dist_vect)/2.0)
            
        for k in range(numattr):    #looping through attributes
            datatype = attr[header[k]][0]
            if(datatype == 'continuous'):
                mmdiff = attr[header[k]][3]
                
            count_hit_near = count_miss_near = 0.0
            count_hit_far = count_miss_far = 0.0
            diff_hit_near = diff_miss_near = 0.0
            diff_hit_far = diff_miss_far = 0.0
            
            class_Store_near = makeClassPairMap()
            class_Store_far = makeClassPairMap()
            
            for i in range(datalen):                     
                xik = x[i][k]
                if(isnan(xik)): continue
                for j in range(i,datalen):
                    xjk = x[j][k]
                    if(i == j or isnan(xjk)): continue
    
                    if(datatype == 'continuous'):
                        calc = abs(xik - xjk) / mmdiff
                        calc1 = (1 - abs(xik - xjk)) / mmdiff
    
                    locator = [i,j]
                    if(i < j): locator.reverse()
                    d = distArray[locator[0]][locator[1]]
                    #--------------------------------------------------------------
                    if (d < (avg_dist[i] - D[i])): #Near
                        if(y[i] == y[j]):
                            count_hit_near += 1
                            if(xik != xjk):
                                if(datatype == 'continuous'):
                                    diff_hit_near -= calc
                                else:
                                    diff_hit_near -= 1
                        else:
                            count_miss_near += 1
                            locator = [y[i],y[j]]
                            if(y[i] < y[j]): locator.reverse()
                            tempString = str(locator[0]) + str(locator[1])
                            class_Store_near[tempString][0] += 1
                            if(xik !=xjk):
                                if(datatype == 'continuous'):
                                    class_Store_near[tempString][1] += calc
                                else:#Discrete
                                    class_Store_near[tempString][1] += 1
                    #--------------------------------------------------------------
                    if (d > (avg_dist[i] + D[i])): #Far
                            
                        if(y[i] == y[j]):
                            count_hit_far += 1
                            if(datatype == 'continuous'):
                                diff_hit_far -= calc1  # Attribute being similar is
                            else:                      # more important.
                                if(xik == xjk):
                                    diff_hit_far -= 1
                        else:
                            count_miss_far += 1
                            locator = [y[i],y[j]]
                            if(y[i] < y[j]): locator.reverse()
                            tempString = str(locator[0]) + str(locator[1])
                            class_Store_far[tempString][0] += 1
                            
                            if(datatype == 'continuous'):
                                class_Store_far[tempString][1] += calc 
                            else:
                                if(xik == xjk):
                                    class_Store_far[tempString][1] += 1    
            #Near
            missSum = 0 
            for each in class_Store_near:
                missSum += class_Store_near[each][0]
                             
            hit_proportion = count_hit_near/float(count_hit_near+count_miss_near) 
            miss_proportion = count_miss_near/float(count_hit_near+count_miss_near) 
            
            for each in class_Store_near:
                diff_miss_near += \
                (class_Store_near[each][0]/float(missSum))*class_Store_near[each][1]
            diff_miss_near = diff_miss_near * float(len(class_Store_near))
    
            diff = diff_miss_near*hit_proportion + diff_hit_near*miss_proportion 
                             
            #Far
            missSum = 0 
            for each in class_Store_far:
                missSum += class_Store_far[each][0]
    
            hit_proportion = count_hit_far/float(count_hit_far + count_miss_far)
            miss_proportion = count_miss_far/float(count_hit_far + count_miss_far) 
            
            for each in class_Store_far:
                diff_miss_far += \
                (class_Store_far[each][0]/float(missSum))*class_Store_far[each][1]
    
            diff_miss_far = diff_miss_far * float(len(class_Store_far))
            
            diff += diff_miss_far*hit_proportion + diff_hit_far*miss_proportion   
               
            ScoreList[k] += diff    
            
        return ScoreList
