
import sys
import os
import argparse
import time
from random import shuffle



scoreDict = {}
for i in range(0,1):
    #Open target CV dataset
    output = "outputs/epifeaturesDataiter_additive__Train.txt"
    #*********************************************************************        
    try:
        f = open(output, 'r')  # opens each datafile to read.
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
    print(scoreDict)