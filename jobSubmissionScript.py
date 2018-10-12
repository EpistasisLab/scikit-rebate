import sys
import os
import argparse
import time


def main(argv):

    
    writepath = outputPath+coreDataName

    dataSize = [100,1000,10000,100000]
    algorithms = ["turf", "vls", "iter"]

    for i in dataSize:
        for algorithm in algorithms:
            #for j in range(1,30):
            j = 1
            dataset = projectDataPath+dataPath+'a_'+str(i) + '/' +numInstances +'/'+heritability+'/'+'a_'+str(i)+numInstances+heritability+'_EDM-2/'+'a_'+str(i)+numInstances+heritability+'_EDM-2'+'_'+str(j).zfill(2)+'.txt.gz'
            outfile = writepath+algorithm+ '_' +coreDataName + '_' + str(i)+'_Train.txt'

            jobName = scratchPath+'scikit-rebate_'+coreDataName+'_'+str(algorithm)+str(i)+'_'+str(time.time())+'_run.sh'                                                  
            shFile = open(jobName, 'w')
            shFile.write('#!/bin/bash\n')

            shFile.write('#BSUB -J '+'scikit-rebate_'+coreDataName+'_'+str(algorithm)+str(i)+'_'+str(time.time())+'\n')
            #shFile.write('#BSUB -M 45000'+'\n')
            shFile.write('#BSUB -o ' + logPath+'scikit-rebate_'+coreDataName+'_'+str(algorithm)+str(i)+'_'+str(time.time())+'.o\n')
            shFile.write('#BSUB -e ' + logPath+'scikit-rebate_'+coreDataName+'_'+str(algorithm)+str(i)+'_'+str(time.time())+'.e\n')

            if i > 10000:
                shFile.write('#BSUB -R "rusage[mem=12000]"\n')
                shFile.write('#BSUB -M 15000\n\n')
                
            shFile.write('python '+'run_job_sub.py '+str(dataset)+' '+str(outfile)+' '+str(algorithm)+'\n') 
            print(str(dataset)+' '+str(outfile)+' '+str(algorithm))

            shFile.close()
            if i > 10000:
                os.system('bsub -q moore_long < ' + jobName)
            else:
                os.system('bsub < '+jobName)   

if __name__=="__main__":
    

    #Identify data path
    projectDataPath = '/home/alexmxu/'
    heritability = 'her_0.4__maf_0.2'

    dataPath = 'datasets/'
    numInstances = 's_1600'

    coreDataName = 'epifeaturesData'

    CVCount = 10
    rebatePath = '/home/alexmxu/scikit-rebate/skrebate'
    outputPath = '/home/alexmxu/outputs/'
    scratchPath = '/home/alexmxu/scratch/'
    logPath = '/home/alexmxu/logs/'
    
    sys.exit(main(sys.argv))

