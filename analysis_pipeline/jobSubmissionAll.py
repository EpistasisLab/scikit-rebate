import sys
import os
import argparse
import time

def main(argv):
    writepath = outputPath+coreDataName
    data_type = ['additive', 'continuous', 'heterogeneous', 'imbalanced', 'na']
    data_size = [100,1000,10000,100000]
    algorithms = ["turf", "vls", "iter"] 
    
    for dirpath, dirnames, filenames in os.walk(dataPath):
        # In some directories, there are report files. Skip these files.
        if len(filenames) < 10 and len(dirpath) > 0:
            continue
        for filename in filenames:
            if not filename.endswith('.txt.gz'):
                continue
            for algorithm in algorithms:
                dataset = os.path.join(dirpath, filename)
                #dataname = dataset.split('/home/alexmxu/datasets/Simulated_Benchmark_Archive')
                outfile = outputPath + '/' + filename +str(algorithm)+'_' +'_Train.txt'

                jobName = scratchPath+'scikit-rebate_'+filename +'_'+str(algorithm)+'_'+str(time.time())+'_run.sh'                                                  
                shFile = open(jobName, 'w')
                shFile.write('#!/bin/bash\n')

                shFile.write('#BSUB -J '+'scikit-rebate_'+filename+'_'+str(algorithm)+'_'+str(time.time())+'\n')
                #shFile.write('#BSUB -M 45000'+'\n')
                shFile.write('#BSUB -o ' + logPath+'scikit-rebate_'+filename+'_'+str(algorithm)+'_'+str(time.time())+'.o\n')
                shFile.write('#BSUB -e ' + logPath+'scikit-rebate_'+filename+'_'+str(algorithm)+'_'+str(time.time())+'.e\n')

                '''if i > 10000:
                    shFile.write('#BSUB -R "rusage[mem=12000]"\n')
                    shFile.write('#BSUB -M 15000\n\n')'''
                    
                shFile.write('python '+'run_job_sub.py '+str(dataset)+' '+str(outfile)+' '+str(algorithm)+'\n') 
                print(str(dataset)+' '+str(outfile)+' '+str(algorithm))

                shFile.close()
                if "a_100000" in jobName:
                    os.system('bsub -q moore_long < ' + jobName)
                else:
                    os.system('bsub < '+jobName)   

if __name__=="__main__":
    

    #Identify data path
    projectDataPath = '/home/alexmxu/'
    heritability = 'her_0.4__maf_0.2'

    dataPath = '/home/alexmxu/datasets/Simulated_Benchmark_Archive'
    numInstances = 's_1600'

    coreDataName = ''

    CVCount = 10
    rebatePath = '/home/alexmxu/scikit-rebate/skrebate'
    outputPath = '/home/alexmxu/outputs/'
    scratchPath = '/home/alexmxu/scratch/'
    logPath = '/home/alexmxu/logs/'
    
    sys.exit(main(sys.argv))

