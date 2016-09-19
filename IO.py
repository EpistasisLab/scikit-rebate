"""
Mon Sep 19 09:57:22 EDT 2016
"""
###############################################################################
# Functions: getArguments, read_data, create_subset, createScoresFile
###############################################################################
import argparse
import time as tm
import numpy as np
import pandas as pd
import sys
import os

def getArguments():
    """get all command line arguments here"""
    options = dict()

    parser = argparse.ArgumentParser(description = \
        "Run ReliefF/MultiSURF/SURF/SURF* on your data")

    parser.add_argument("-a", "--algorithm", type=str, \
        help="relieff, multisurf, surf, surfstar (default=relieff)")
    parser.add_argument("-c", "--classname", type=str, \
        help="name of Class/Phenotype column (default=Class)")
    parser.add_argument("-D", "--debug", \
        help="lots and lots of output (not implemented yet)", action="store_true")
    parser.add_argument("-d", "--discretelimit", type=int, \
        help="max unique values in attributes/class to be considered \
              discrete (default=10)")
    parser.add_argument("-f", "--filename", type=str, \
        help="name of training data file (REQUIRED)")
    parser.add_argument("-k", "--knearestneighbors", type=int, \
        help="k number of neighbors for ReliefF to use (default=10)")
    parser.add_argument("-m", "--missingdata", type=str, \
        help="missing data designator or string (default=NA)")
    parser.add_argument("-o", "--outputdir", type=str, \
        help="directory path to write scores file (default=data file directory)")
    parser.add_argument("-T", "--topattr", type=int, \
        help="Create new data file with top number attributes (integer argument)")
    parser.add_argument("-t", "--turflimit", type=int, \
        help="percent_drop (default=0: turf OFF )")
    parser.add_argument("-v", "--verbose", \
        help="use output verbosity", action="store_true")
    parser.add_argument("-x", "--testdata", type=str, \
        help="test data file, used in conjuction with --topattr to create new \
              test data file with top number of attributes")
    args = parser.parse_args()
    # ------------------------------------------ #
    if(args.filename ==  None):
        print("filename required!")
        sys.exit()
    else:
        options['filename'] = args.filename
        options['basename'] = os.path.basename(args.filename)
        options['dir_path'] = os.path.dirname(args.filename)
    # ------------------------------------------ #
    if(args.testdata ==  None):
        options['testdata'] = None
    else:
        options['testdata'] = args.testdata
        options['test_basename'] = os.path.basename(args.testdata)
        options['test_dir_path'] = os.path.dirname(args.testdata)
    # ------------------------------------------ #
    if(args.classname == None):
        phenoTypeName = "Class"
    else:
        phenoTypeName = args.classname
    options['phenotypename'] = phenoTypeName
    # ------------------------------------------ #
    if(args.discretelimit == None):
        discretelimit = 10
    else:
        discretelimit = args.discretelimit
    options['discretelimit'] = discretelimit
    # ------------------------------------------ #
    if(args.knearestneighbors == None):
        neighbors = 10
    else:
        neighbors = args.knearestneighbors
    options['neighbors'] = neighbors
    # ------------------------------------------ #
    if(args.missingdata == None):
        mdata = 'NA'
    else:
        mdata = args.missingdata
    options['missingdata'] = mdata
    # ------------------------------------------ #
    if(args.algorithm == None):
        algorithm = 'relieff'
    else:
        algorithm = args.algorithm
    options['algorithm'] = algorithm
    # ------------------------------------------ #
    if(args.turflimit == None):
        turf = '0'
    else:
        turf = args.turflimit
    options['turfpct'] = turf
    # ------------------------------------------ #
    if(args.verbose):
        V = True
    else:
        V = False
    options['verbose'] = V
    # ------------------------------------------ #
    if(args.debug):
        D = True
    else:
        D = False
    options['debug'] = D
    # ------------------------------------------ #
    if(args.topattr == None):
        topattr = 0
    else:
        topattr = args.topattr
    options['topattr'] = topattr
    # ------------------------------------------ #
    if(args.outputdir == None):
        outputdir = '.'
    else:
        outputdir = args.outputdir
    options['outputdir'] = outputdir
    # ------------------------------------------ #

    return options
###############################################################################
def test_testdata(header, testdata, options):
    """ ensure the test data has the same attributes 
        and class as the training data"""

    theader, tdata = read_data(testdata, options)

    for i in header:
        if(i not in tdata.columns):
            print("Features must match between training and test data")
            sys.exit(3)

    for i in tdata.columns:
        if(i not in header):
            print("Features must match between training and test data")
            sys.exit(3)

    return theader, tdata
###############################################################################
def create_subset(header, x, y, options, ordered_attr):
    """ creates the a subset of top attributes of the  training data file"""
    V = options['verbose']
    top = []
    topidx = []  # index of columns in x to extract
    outfile = options['basename']
    path = options['outputdir']
    dir_path = options['dir_path']
    if(path == '.' and dir_path != ''):
        path = options['dir_path']

    outfile = path + '/top_' + str(options['topattr']) + '_attrs-' + outfile
    for i in range(options['topattr']):
        topidx.append(header.index(ordered_attr[i]))
        top.append(ordered_attr[i])

    top.append(options['phenotypename'])
    newx = x[:,topidx]
    newy = y.reshape(len(y),1)
    npdata = np.append(newx,newy,axis=1)
    newdata = pd.DataFrame(npdata,columns=top)

    fh = open(outfile, 'w')
    newdata.to_csv(fh, sep='\t', index=False)
    if(V):
        ctime = "[" + tm.strftime("%H:%M:%S") + "]"
        print(ctime + " Created new data file: " + outfile)
        sys.stdout.flush()
###############################################################################
def create_test_subset(tdata, options, ordered_attr):
    """creates the same subset of top attributes of the testdata file as the
       training data"""
    V = options['verbose']
    top = []
    outfile = options['test_basename']
    path = options['outputdir']
    dir_path = options['test_dir_path']
    if(path == '.' and dir_path != ''):
        path = options['test_dir_path']

    outfile = path + '/top_' + str(options['topattr']) + '_attrs-' + outfile
    for i in range(options['topattr']):
        top.append(ordered_attr[i])

    top.append(options['phenotypename'])
    newdata = tdata[top]

    fh = open(outfile, 'w')
    newdata.to_csv(fh, sep='\t', index=False)
    if(V):
        ctime = "[" + tm.strftime("%H:%M:%S") + "]"
        print(ctime + " Created new test data file: " + outfile)
        sys.stdout.flush()
###############################################################################
def createScoresFile(header,var,scores,options,prog_start,turfpct,table,lost):
    from operator import itemgetter

    V = options['verbose']
    input_file = options['basename']
    algorithm = options['algorithm']
    path = options['outputdir']
    dir_path = options['dir_path']
    if(path == '.' and dir_path != ''):
        path = options['dir_path']

    tab = '\t'; nl = '\n'
    top = []

    if('turf' not in algorithm):
        table = []
        for i in range(var['NumAttributes']):
            table.append((header[i], scores[i]))
        table = sorted(table,key=itemgetter(1), reverse=True)

    if('relieff' in algorithm):
        values = str(var['discreteLimit']) + '-' + str(var['numNeighbors'])
    elif('surf' in algorithm):
        values = str(var['discreteLimit'])
    if('turf' in algorithm):
        values += '-' + str(turfpct)

    outfile = path + '/' + algorithm + '-scores-' + values + '-' + input_file
    fh = open(outfile, 'w')
    fh.write(algorithm + ' Analysis Completed with REBATE\n')
    fh.write('Run Time (sec): ' + str(tm.time() - prog_start) + '\n')
    fh.write('=== SCORES ===\n')

    n = 1
    if('turf' in algorithm):
        for col, val in table:
            top.append(col)
            val = '{0:.16f}'.format(val)
            fh.write(col + tab + str(val) + tab + str(n) + nl)
            n += 1

        reduction = 0.01 * (max(scores) - min(scores))
        m = last = 0
        for w in sorted(lost, key=lost.get, reverse=True):
            if(last != lost[w]):
                last = lost[w]
                m += 1
            top.append(w)
            score = min(scores) - reduction * m
            score = '{0:.16f}'.format(score)
            fh.write(w + tab + str(score) + tab + str(lost[w] * '*' + nl))
    else: # NOT TURF
        for col, val in table:
            top.append(col)
            val = '{0:.16f}'.format(val)
            fh.write(col + tab + str(val) + tab + str(n) + nl)
            n += 1
    fh.close()

    if(V):
        ctime = "[" + tm.strftime("%H:%M:%S") + "]"
        print(ctime + " Created scores file: " + outfile)
        sys.stdout.flush()

    return top
###############################################################################
def printf(format, *args):
    sys.stdout.write(format % args)
###############################################################################
def np_read_data(fname, options):
    """Read in data file into a numpy array (data) and a header
       returns header, data in that order."""

    import csv

    start = tm.time()
    V = options['verbose']
    md = options['missingdata']

    #---- determine delimiter  -----------#
    fh = open(fname)
    line = fh.readline().rstrip()
    fh.close()
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(line)
    delim = dialect.delimiter
    #-------------------------------------#

    # reading into numpy array
    data = np.genfromtxt(fname, missing_values=md, skip_header=1,
                         dtype=np.double, delimiter=delim)
    if(V):
        ctime = "[" + tm.strftime("%H:%M:%S") + "]"
        print(ctime + " " + fname + ": data input elapsed time(sec) = " 
                    + str(tm.time() - start))
        sys.stdout.flush()

    #delim = '"' + delim + '"'
    header = line.split(delim)

    return header, data
###############################################################################
def getxy(header, data, options):
    """ returns contiguous x numpy matrix of data and y numpy array of class
        and also removes phenotype name from headers to match the columns in
        the x matrix"""
    pname = options['phenotypename']
    pindex = header.index(pname)

    y = data[:, pindex]
    y = np.ascontiguousarray(y, dtype=np.double)
    x = np.delete(data,pindex,axis=1)
    x = np.ascontiguousarray(x, dtype=np.double)
    options['classloc'] = pindex
    del header[pindex]  # remove phenotype/class name from header

    return x, y
