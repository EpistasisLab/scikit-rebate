import sys
import os
import argparse
import time


def main(argv):

    
    writepath = outputPath+coreDataName
    data_type = ['additive', 'continuous', 'heterogeneous', 'imbalanced', 'na']
    data_size = [100,1000,10000,100000]
    algorithms = ["turf", "vls", "iter"] 
    
    for algorithm in algorithms:
        for datat in data_type:
            #for j in range(1,30):
            j = 1
            i=1
            data_dir = projectDataPath + dataPath + datat +'/'
            for dir_name, subdir_list, file_list in os.walk(data_dir):
                for file in file_list:
                    dataset = data_dir+file
                    outfile = writepath+algorithm+ '_' +datat + '_' +'_Train.txt'

                    jobName = scratchPath+'scikit-rebate_'+datat+'_'+str(algorithm)+str(i)+'_'+str(time.time())+'_run.sh'                                                  
                    shFile = open(jobName, 'w')
                    shFile.write('#!/bin/bash\n')

                    shFile.write('#BSUB -J '+'scikit-rebate_'+datat+'_'+str(algorithm)+str(i)+'_'+str(time.time())+'\n')
                    #shFile.write('#BSUB -M 45000'+'\n')
                    shFile.write('#BSUB -o ' + logPath+'scikit-rebate_'+datat+'_'+str(algorithm)+str(i)+'_'+str(time.time())+'.o\n')
                    shFile.write('#BSUB -e ' + logPath+'scikit-rebate_'+datat+'_'+str(algorithm)+str(i)+'_'+str(time.time())+'.e\n')

                    '''if i > 10000:
                        shFile.write('#BSUB -R "rusage[mem=12000]"\n')
                        shFile.write('#BSUB -M 15000\n\n')'''
                        
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

    coreDataName = ''

    CVCount = 10
    rebatePath = '/home/alexmxu/scikit-rebate/skrebate'
    outputPath = '/home/alexmxu/outputs/'
    scratchPath = '/home/alexmxu/scratch/'
    logPath = '/home/alexmxu/logs/'
    
    sys.exit(main(sys.argv))

