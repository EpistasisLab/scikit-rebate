import sys
import os
import argparse
import time

def main(argv):

	algorithms = ['turf', 'vls', 'iter']
	dataSize = [100,1000]
	writepath = '/mnt/c/Users/Alex/cs/python/scikit-rebate/outputs/epifeaturesData'

	for algorithm in algorithms:
		for data in dataSize:
			out_name = writepath+algorithm+ '_' +coreDataName + '_' + str(data)+'_Train.txt'
			print(parse_output(out_name))


def parse_output(output_filename):
    start_parsing = False
    filename = ''
    algorithm = ''
    scoring_time = ''
    variables_ranked = []
    variables_scores = []
    index1 = 0
    index2 = 0
    
    with open(output_filename) as in_file:
        for line in in_file:
            if 'iter' in line:
                algorithm += 'iter'
            elif 'vls' in line:
                algorithm += 'vls'
            elif 'turf' in line:
                algorithm += 'turf'
            elif 'Run Time' in line:
                    scoring_time = line.split('Completed scoring in')[1].split('seconds')[0].strip()
            elif line.count('\t') == 1:
                variables_ranked.append(line.split('\t')[0].strip())
                variables_scores.append(line.split('\t')[1].strip())

    for variable in variables_ranked:
    	if variable == 'M0P0'
    		index1 = variables_ranked.index(variable)
    	if variable == 'M0P1'
    		index2 = variables_ranked.index(variable)

    return filename, algorithm, scoring_time, ','.join(variables_ranked), ','.join(variables_scores)

if __name__=="__main__":
    

    #Identify data path
    projectDataPath = '/home/alexmxu/'
    heritability = 'her_0.4__maf_0.2'

    dataPath = 'datasets/'
    numInstances = 's_1600'

    coreDataName = 'epifeaturesData'

    rebatePath = '/home/alexmxu/scikit-rebate/skrebate'
    outputPath = '../outputs/'
    scratchPath = '/home/alexmxu/scratch/'
    logPath = '/home/alexmxu/logs/'
    
    sys.exit(main(sys.argv))