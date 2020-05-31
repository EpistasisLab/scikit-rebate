
import sys
import os
import argparse
import time
from random import shuffle
from scipy import stats


from glob import glob
from tqdm import tqdm
import gzip

def parse_output(output_filename):
    start_parsing = False
    filename = ''
    algorithm = ''
    scoring_time = ''
    variables_scores = []
    variables_ranked = []
    
    with open(output_filename) as in_file:
        for line in in_file:
            if '/home/alexmxu/datasets/' in line:
                filename = line.strip()
            elif 'iter' in line or 'turf' in line or 'vls' in line or 'multisurf' in line:
                algorithm = line.strip()
            elif 'Run Time (sec)' in line:
                scoring_time = line.split('Run Time (sec):')[1].strip()
            elif line.count('\t') >= 1:
                variables_ranked.append(line.split('\t')[0].strip())
                variables_scores.append(line.split('\t')[1].strip())
    return filename, algorithm, scoring_time, ','.join(variables_ranked), ','.join(variables_scores)

with gzip.open('benchmark-parsed.tsv.gz', 'w') as out_file:
    out_text = '\t'.join(['FileName', 'Algorithm', 'RunTimeSecs', 'FeaturesRanked', 'FeatureScores']) + '\n'
    out_file.write(out_text.encode('UTF-8'))
    for filename in tqdm(sorted(glob('outputs/*'))):
        #if 'a_10000' in filename:
        #    continue
        parsed_output = parse_output(filename)
        #if '/project/moore/users/ryanurb/Simulated_Benchmark_Archive/Simulated_Benchmark_Archive/GAMETES_2.2_dev_peter_mainEff_additive_4_Datasets_2Het_Loc_1_Qnt_2_Pop_100000/a_20/s_1600/h_0.4MAF_0.2/r_50/a_20s_1600_Het_h_0.4MAF_0.2_r_50_EDM-1' in parsed_output[0]:
        #    continue

        parser_error = False
        for output in parsed_output:
            if output == '':
                parser_error = True
                break   
        if parser_error:
            continue
        out_text = '\t'.join(parsed_output) + '\n'
        out_file.write(out_text.encode('UTF-8'))



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
