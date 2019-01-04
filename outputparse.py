
import sys
import os
import argparse
import time
from random import shuffle
from scipy import stats

'''for root, dirs, outputs in os.walk("outputs/"):
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

'''
def parse_output(output_filename):
    start_parsing = False
    filename = ''
    algorithm = ''
    scoring_time = ''
    variables_names = []
    variables_scores = []
    varaibles_ranked = []
    
    with open(output_filename) as in_file:
        for line in in_file:
            if 'The output (if any) follows:' in line:
                start_parsing = True
                continue
            
            if start_parsing:
                if '/project/moore/users/' in line:
                    filename = line.strip()
                elif 'n_features_to_select=' in line:
                    algorithm = line.strip()
                elif 'verbose=True' in line:
                    algorithm += ' ' + line.strip()
                elif 'ANOVAFValue' in line or line.strip() == 'chi2' or 'MutualInformation' in line or 'ExtraTrees' in line:
                    algorithm = line.strip()
                elif 'Completed scoring' in line:
                    scoring_time = line.split('Completed scoring in')[1].split('seconds')[0].strip()
                elif line.count('\t') == 1:
                    variables_ranked.append(line.split('\t')[0].strip())
                    variables_scores.append(line.split('\t')[1].strip())
    return filename, algorithm, scoring_time, ','.join(variables_ranked), ','.join(variables_scores)