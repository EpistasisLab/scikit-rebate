
import sys
import os
import argparse
import time
from random import shuffle
from scipy import stats

for root, dirs, outputs in os.walk("outputs/"):
    for output in outputs:
        scoreDict = {}
        for i in range(0,1):
            #Open target CV dataset
            #output = "outputs/epifeaturesDataiter_additive__Train.txt"
            #*********************************************************************        
            try:
                f = open('outputs/' + output, 'r')  # opens each datafile to read.
            except:
                print("Data-set Not Found! Progressing without it.")
                print(output)
                continue
            #*********************************************************************  
            counter = 0
            for line in f:
                if counter < 3: #ignore the first three rows (Relief-file format)
                    pass
                else:
                    if i == 0:
                        tempList = line.strip().split('\t')
                        scoreDict[tempList[0]] = float(tempList[1]) #all attributes and corresponding EK scores are hashed in a dictionary
                    else: 
                        tempList = line.strip().split('\t')
                        scoreDict[tempList[0]] += float(tempList[1]) #all attributes and corresponding EK scores are hashed in a dictionary
                counter += 1
            f.close()
        sortedScoreDict = sorted(scoreDict.values(), reverse = True)
        #print(sortedScoreDict)

        alldatagood = True
        for item in scoreDict:
            if(item[0] == 'M'):
                #print(item)
                if (stats.percentileofscore(sortedScoreDict, scoreDict[item]) > 80):
                    pass
                else:
                    print(stats.percentileofscore(sortedScoreDict, scoreDict[item]) + 'percentile less than threshold')
                    alldatagood = False
        print("data looks " + str(alldatagood)) 
        #print(scoreDict)