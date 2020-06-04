import sys
import os
import argparse

'''
Sample Run Code
python checkjobs --data-path /Users/robert/Desktop/rebateDatasets --output-path /Users/robert/Desktop/outputs --experiment-name rebate1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-path', dest='data_path', type=str, help='path to directory containing datasets')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name

    # Check to make sure data_path exists and experiment name is valid & unique
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    if not os.path.exists(output_path+'/'+experiment_name):
        raise Exception("Provided experiment path does not exist")

    all_done = True
    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            if not filename.endswith('.txt.gz'):
                continue
            for algorithm in ['multisurf','vls','iter','turf','vls_iter','vls_turf']:
                outfile = output_path + '/' + experiment_name + '/rawoutputs/' + algorithm + '_' + filename[:-3]
                if not os.path.exists(outfile):
                    all_done = False
                    print(algorithm+' on '+filename[:-3]+' not complete')

    if all_done:
        print("All jobs complete")

if __name__ == '__main__':
    sys.exit(main(sys.argv))