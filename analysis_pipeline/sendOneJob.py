import sys
import os
import argparse
import jobSubmissionMain

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-path', dest='data_path', type=str, help='path to dataset file')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--algo', dest='algo', type=str, help='multisurf,vls,iter,turf,vls_iter,vls_turf')
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets',default="Class")
    parser.add_argument('--random-state', dest='random_state', type=int, default=42)

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name
    algorithm = options.algo
    class_label = options.class_label
    random_state = options.random_state

    # Check to make sure data_path exists and experiment name is valid & unique
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Provided experiment path does not exist")

    filename = data_path.split('/')[-1]
    outfile = output_path + '/' + experiment_name + '/rawoutputs/' + algorithm + '_' + filename[:-3]
    #jobSubmissionMain.submitLocalJob(algorithm,data_path,class_label,random_state,outfile)
    jobSubmissionMain.submitClusterJob(algorithm, data_path, output_path + '/' + experiment_name, class_label,random_state, outfile)

if __name__ == '__main__':
    sys.exit(main(sys.argv))