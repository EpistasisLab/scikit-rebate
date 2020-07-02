import sys
import os
import argparse
import time
import jobSubmission

'''
Sample Run Code:
python jobSubmissionMain.py --data-path /home/robertzh/data --output-path /home/robertzh/outputs --experiment-name run2
python jobSubmissionMain.py --data-path /Users/yesuyu/Desktop/ReliefMultiset2 --output-path /Users/yesuyu/Desktop/outputs --experiment-name ReliefMultiset2

'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-path', dest='data_path', type=str, help='path to directory containing datasets')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets',default="Class")
    parser.add_argument('--random-state', dest='random_state', type=int, default=42)

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name
    class_label = options.class_label
    random_state = options.random_state

    # Check to make sure data_path exists and experiment name is valid & unique
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890':
            raise Exception('Experiment Name must be alphanumeric')

    # Create output folder if it doesn't already exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Create Experiment folder, with log and job folders
    if not os.path.exists(output_path + '/' + experiment_name):
        os.mkdir(output_path + '/' + experiment_name)
    if not os.path.exists(output_path + '/' + experiment_name + '/jobs'):
        os.mkdir(output_path + '/' + experiment_name + '/jobs')
    if not os.path.exists(output_path + '/' + experiment_name + '/logs'):
        os.mkdir(output_path + '/' + experiment_name + '/logs')

    #CONTROL PANEL######################################################################################################
    #Choose from 'multisurf','vls','iter','turf','vls_iter','vls_turf','multisurf_abs','vls_abs','iter_abs','turf_abs','vls_iter_abs','vls_turf_abs','relieff_100nn','relieff_100nn_abs'

    algorithms_to_use = ['relieff_100nn','relieff_100nn_abs','multisurf','multisurf_abs']

    ####################################################################################################################

    #Iterate through directories
    if not os.path.exists(output_path + '/' + experiment_name + '/rawoutputs'):
        os.mkdir(output_path + '/' + experiment_name + '/rawoutputs')
    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            if not filename.endswith('.txt.gz'):
                continue
            for algorithm in algorithms_to_use:
                outfile = output_path + '/' + experiment_name + '/rawoutputs/' + algorithm + '_' + filename[:-3]
                submitLocalJob(algorithm,os.path.join(dirpath, filename),class_label,random_state,outfile)
                #submitClusterJob(algorithm, os.path.join(dirpath, filename), output_path + '/' + experiment_name,class_label, random_state,outfile)

def submitLocalJob(algorithm,datapath,class_label,random_state,outfile):
    jobSubmission.job(algorithm,datapath,class_label,random_state,outfile)

def submitClusterJob(algorithm,datapath,experiment_path,class_label,random_state,outfile):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/jobSubmission.py ' + algorithm + " " + datapath + " " + class_label +
                  ' ' + str(random_state) + ' ' + outfile + '\n')
    sh_file.close()
    if 'a_100000' in datapath:
        os.system('bsub -q moore_long < ' + job_name)
    else:
        os.system('bsub < ' + job_name)

if __name__ == '__main__':
    sys.exit(main(sys.argv))