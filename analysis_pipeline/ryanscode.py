"""
Name:        sum_scikit-rebate.py
Authors:     Ryan Urbanowicz - Written at the University of Pennsylvania, Philadelphia, PA
Contact:     ryanurb@upenn.edu
Created:     10/20/18
Description: This script creates a single summary rebate output file that averages the CV run scores and uses either or both permutation testing runs
            to generate p-values and identify significance over rebate feature scores. 
Dependencies: This script 
Example: 
python ./sum_scikit-rebate.py -o /home/ryanurb/idata/output/MultiSURF_EpiOnly_20180710_Clean -p -d
python ./sum_scikit-rebate.py -o /home/ryanurb/idata/output/MultiSURF_Epi_DietAdj_20180710_Clean -p -d
python ./sum_scikit-rebate.py -o /home/ryanurb/idata/output/MultiSURF_Epi_DietAdj_Matched_20180710_Clean -p -d
python ./sum_scikit-rebate.py -o /home/ryanurb/idata/output/MultiSURF_Epi_DietAdj_Matched_20180710_Clean_CV_M -p -d -f CV_M -b MultiSURF_Epi_DietAdj_Matched_20180710_Clean
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Script to run 10-fold CV analysis of ReBATE for Lynch collaboration

import sys
import os
import argparse
import time
from random import shuffle

def main(argv):
   #Argument Parsing :------------------------------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Create CV partitioned datasets and/or datasets with a permuted endpoint.')
    parser.add_argument('-o', '--outputpath', help='Str: Base folder path for any output files', type=str, default='')
    parser.add_argument('-t', '--nameModifier', help='Str: Additional text for output folder beyond algorithm and dataset name', type=str, default='')
    parser.add_argument('-k', '--folds', help='Int: number of CV partitions/folds', type=int, default=10)
    parser.add_argument('-n', '--permutations', help='Int: number of CV partitions/folds', type=int, default=100)
    parser.add_argument('-f', '--cvfolder', help='Str: specifies the name of the folder for the specific type of cv used.', type=str, default='CV_S')
    parser.add_argument('-b', '--basefilename', help='Str: Whenever the base folder does not have same name as files specify a different base name.', type=str, default='None')
    
    parser.add_argument('-a', '--algorithm', help='Str: ReBATE algorithm to apply (i.e. ReliefF, SURF, SURFstar, MultiSURF, or MultiSURFstar', type=str, default='MultiSURF')
    parser.add_argument('-p','--dopermstats', help='Boolean: Determine p-values based on permutation runs', action='store_true')
    parser.add_argument('-d','--docvpermstats', help='Boolean: Determine p-values based on CV-permutation runs', action='store_true')
 
    parser.add_argument('-z', '--scratchpath', help='Str: specifies path to the cluster scratch folder. This is where job submission files will be saved.', type=str, default='/home/ryanurb/idata/scratch')
    parser.add_argument('-y', '--logpath', help='Str: specifies path to the cluster log folder. This is where standard output and error files will be saved.', type=str, default='/home/ryanurb/idata/logs')
    parser.add_argument('-x', '--runpath', help='Str: specifies path for the run_scikit-rebate script.', type=str, default='/home/ryanurb/lynch_project/RBA_LCS_pipeline')
    
    args = parser.parse_args()

    #Parse and establish names of key files and folders
    if args.basefilename == 'None': #No name change
        dataname = parseName(args.outputpath)
    else:
        dataname = args.basefilename
    
    #Create Feature Score Dictionary to sum feature scores over each CV file --------------------------------------------------------
    scoreDict = {}
    for i in range(0,args.folds):
        #Open target CV dataset
        output = args.outputpath+'/'+args.cvfolder+'/'+dataname+'_'+args.cvfolder+'_'+str(i)+'_Train.txt'
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
            
    #Make the sum of scores an average
    for i in scoreDict:
        scoreDict[i] = scoreDict[i]/float(args.folds)
        
    if args.dopermstats: # Use originial permuted datasets to derive p-values
        permuteScoreDict = {}
        for i in range(0,args.permutations):
            #Open target CV dataset
            output = args.outputpath+'/P/'+dataname+'_P_'+str(i)+'_Train.txt'
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
                        permuteScoreDict[tempList[0]] = [float(tempList[1])] #all attributes and corresponding EK scores are hashed in a dictionary
                    else: 
                        tempList = line.strip().split('\t')
                        permuteScoreDict[tempList[0]].append(float(tempList[1])) #all attributes and corresponding EK scores are hashed in a dictionary
                counter += 1
            f.close()
        
        pvalDict = {}
        for each in scoreDict:
            #each feature name
            permuteList = sorted(permuteScoreDict[each])
            real = scoreDict[each]
            #Get pvalue 
            pvalResult = PValCheck(real, permuteList, args.permutations)
            pvalDict[each] = pvalResult
            
            
    if args.docvpermstats: # Use cv permuted datasets to detremine p-values
        permuteScoreCVDict = {}
        for i in range(0,args.permutations):
            tempCount = 0
            for j in range(0,args.folds):            
                #Open target CV dataset
                output = args.outputpath+'/P/'+args.cvfolder+'/'+dataname+'_P_'+str(i)+'_'+args.cvfolder+'_'+str(j)+'_Train.txt'
                #*********************************************************************        
                try:
                    f = open(output, 'r')  # opens each datafile to read.
                    tempCount += 1
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
                        if j ==0: #First CV
                            if i == 0:
                                tempList = line.strip().split('\t')
                                permuteScoreCVDict[tempList[0]] = [float(tempList[1])] #all attributes and corresponding EK scores are hashed in a dictionary
                            else: 
                                tempList = line.strip().split('\t')
                                permuteScoreCVDict[tempList[0]].append(float(tempList[1])) #all attributes and corresponding EK scores are hashed in a dictionary
                        else: # Remaining CV

                            tempList = line.strip().split('\t')
                            permuteScoreCVDict[tempList[0]][i] += float(tempList[1])
                    counter += 1
                f.close()
                
                #Divide each entry by number of CV partitions. 
                for entry in permuteScoreCVDict: #each feature - go through the list of permuted values
                    for v in range(len(permuteScoreCVDict[entry])): #for each perm value in list
                        permuteScoreCVDict[entry][v] = float(permuteScoreCVDict[entry][v]/tempCount)
                
        pvalDict2 = {}
        for each in scoreDict:
            #each feature name
            permuteList = sorted(permuteScoreCVDict[each])
            real = scoreDict[each]
            #Get pvalue 
            pvalResult = PValCheck(real, permuteList, args.permutations)
            pvalDict2[each] = pvalResult      
 
            
    #Make CVAverages output file. ----------------------------------------------------------------------
    outfile = args.outputpath+'/Summary_'+args.algorithm+'_'+dataname+'_'+args.cvfolder+'_Train.txt'

    summaryFile = open(outfile, 'w')
    summaryFile.write(args.algorithm + ' Analysis Completed with REBATE\n')
    summaryFile.write('Run Time (sec): ' + str('NA') + '\n')
    
    if args.dopermstats and args.docvpermstats:
        summaryFile.write('Feature'+'\t'+ 'SCORES'+'\t'+ 'pval-perm'+'\t'+ 'pval-cvperm'+'\n')   
        n = 1
        for key, value in sorted(scoreDict.items(), key=lambda x: x[1], reverse=True): 
            summaryFile.write(str(key) + '\t' + str(value) + '\t' + str(pvalDict[key])+ '\t' + str(pvalDict2[key])+'\n')
            #print("{} : {}".format(key, value))
            n+=1
        summaryFile.close()
    
    elif args.dopermstats and not args.docvpermstats:
        summaryFile.write('Feature'+'\t'+ 'SCORES'+'\t'+ 'pval-perm'+'\n')   
        n = 1
        for key, value in sorted(scoreDict.items(), key=lambda x: x[1], reverse=True): 
            summaryFile.write(str(key) + '\t' + str(value) + '\t' + str(pvalDict[key])+'\n')
            #print("{} : {}".format(key, value))
            n+=1
        summaryFile.close()
        
    elif not args.dopermstats and args.docvpermstats:
        summaryFile.write('Feature'+'\t'+ 'SCORES'+'\t'+ 'pval-cvperm'+'\n')   
        n = 1
        for key, value in sorted(scoreDict.items(), key=lambda x: x[1], reverse=True): 
            summaryFile.write(str(key) + '\t' + str(value) + '\t' + str(pvalDict2[key])+'\n')
            #print("{} : {}".format(key, value))
            n+=1
        summaryFile.close()
    else: #Only scores, no p-balues. 
        summaryFile.write('Feature'+'\t'+ 'SCORES'+'\n')   
        n = 1
        for key, value in sorted(scoreDict.items(), key=lambda x: x[1], reverse=True): 
            summaryFile.write(str(key) + '\t' + str(value) +'\n')
            #print("{} : {}".format(key, value))
            n+=1
        summaryFile.close()
    
def PValCheck(real,permuteList,permutations):
    count = 0
    for i in range(len(permuteList)):
        if real > permuteList[i]:
            count += 1

    if count == permutations:
        pVal = 1/float(permutations)
    else:
        pVal = 1.0 - (count / float(permutations))
    return pVal


def parseName(datapath):
    # Parse the dataset name
    dataname = ''
    dataList = datapath.split('/')
    if len(dataList) > 1: #path given
        dataname = dataList[-1]
    else:
        dataname = dataList[0]

    return dataname
        
if __name__=="__main__":    
    sys.exit(main(sys.argv))