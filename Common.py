"""
Mon Sep 19 09:57:01 EDT 2016
"""
import numpy as np
import math
import pandas as pd
import sys
###############################################################################
def getVariables(header, x, y, options):
    """Get all the needed variables into a Dictionary
       More added in overallDataType()"""

    pname = options['phenotypename']

    var = {'NumAttributes' : len(header),
            'phenoTypeList' : list(set(y)) }

    if(len(var['phenoTypeList']) <= options['discretelimit']):
        var['discretePhenotype'] = True
        var['phenSD'] = 0
    else:
        var['discretePhenotype'] = False
        var['phenSD'] = np.std(y, ddof=1)

    var['discreteLimit'] = options['discretelimit']
    var['labelMissingData'] = options['missingdata']
    var['phenoTypeName'] = pname
    
    #var['phenoTypeLoc'] = options['classloc']
    var['numNeighbors'] = options['neighbors']
    
    var['mdcnt'] = np.isnan(x).sum()
    var['datalen'] = len(x)

    return var

###############################################################################
def getAttributeInfo(header, x, var, options):
    """Get attribute as tuple into Dictionary"""

    attr = dict()

    c = d = 0
    limit = options['discretelimit']
    w = x.transpose()

    #for h in header:
    for idx in range(len(w)):
        h = header[idx]
        z = w[idx]
        z = z[np.logical_not(np.isnan(z))]  # remove missing data before
        zulen = len(np.unique(z))           # calculating unique set
        if(zulen <= limit):
            attr[h] = ('discrete', 0, 0, 0)
            d += 1
        else:
            mx = np.max(z)
            mn = np.min(z)
            attr[h] = ('continuous', mx, mn, mx - mn)
            c += 1

    overallDataType(attr,var,options)  # add datatype of data and endpoints

    var['dpct'] = (float(d) / (d + c) * 100, d)
    var['cpct'] = (float(c) / (d + c) * 100, c)

    return attr

###############################################################################
# Is the data entirely discrete, continuous, or mixed?
# Is the class type discrete, continuous or multiclass?
# This will help with future directions.  Adding this to the variables
# dictionary.  This is called from within getAttributeInfo()
def overallDataType(attr, var, options):

    D = False; C = False # set tmp booleons

    for key in attr.keys():
        if(key == 'dataType' or key == 'phenoType'): continue
        if(attr[key][0] == 'discrete'):
            D = True
        if(attr[key][0] == 'continuous'):
            C = True

    if(D and C):
        dataType = 'mixed'
    elif(D and not C):
        dataType = 'discrete'
    elif(C and not D):
        dataType = 'continuous'

    if(var['discretePhenotype'] and len(var['phenoTypeList']) > 2):
        classtype = 'multiclass'
    elif(var['discretePhenotype']):
        classtype = 'discrete'
    else:
        classtype = 'continuous'

    var['dataType'] = dataType
    var['classType'] = classtype

###############################################################################
def getDistances(x, attr, var):
    """This creates the distance array for only discrete or continuous data with
       no missing data"""
    from scipy.spatial.distance import pdist, squareform
    #--------------------------------------------------------------------------
    def pre_normalize(x):
        idx = 0
        for i in attr:
            cmin = attr[i][2]
            diff = attr[i][3]
            x[idx] -= cmin
            x[idx] /= diff
            idx += 1

        return x
    #--------------------------------------------------------------------------
    dtype = var['dataType']
    numattr = var['NumAttributes']

    if(dtype == 'discrete'):
        return squareform(pdist(x,metric='hamming'))

    else: #(dtype == 'continuous'):
        x = pre_normalize(x)
        return squareform(pdist(x,metric='cityblock'))

###############################################################################
# return mask for discrete(0)/continuous(1) attributes and their indices
# return array of max/min diffs of attributes.
# added for cython routines
def dtypeArray(header, attr, var):
    import numpy as np
    attrtype = []
    attrdiff = []
    pname = var['phenoTypeName']

    for key in header:
        #if(key == pname): continue
        if(attr[key][0] == 'continuous'):
            attrtype.append(1)
        else:
            attrtype.append(0)
        attrdiff.append(attr[key][3])  # build array of max-min diffs

    attrtype = np.array(attrtype)
    cidx = np.where(attrtype == 1)[0]   # grab indices for split_data()
    cidx = np.ascontiguousarray(cidx, dtype=np.int32)
    didx = np.where(attrtype == 0)[0]   # where returns a tuple
    didx = np.ascontiguousarray(didx, dtype=np.int32)
    
    attrdiff = np.array(attrdiff)
    attrdiff = np.ascontiguousarray(attrdiff, dtype=np.double)
    return attrdiff, cidx, didx

###############################################################################
def printf(format, *args):
    sys.stdout.write(format % args)
