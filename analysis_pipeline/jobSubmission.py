import sys
import pandas as pd
import time
from skrebatewip import MultiSURF,TURF,VLS,Iter
import random
import numpy as np

def job(algorithm, datapath, class_label,random_state,outfile):

    #Start Job Timer
    job_start_time = time.time()

    # Unpack data
    genetic_data = pd.read_csv(datapath, sep='\t', compression='gzip')
    features, labels = genetic_data.drop('Class', axis=1).values, genetic_data[class_label].values

    # Train
    if algorithm == 'multisurf':
        estimator = MultiSURF(n_jobs=-1)
    elif algorithm == 'vls':
        estimator = VLS(MultiSURF(n_jobs=-1))
    elif algorithm == 'iter':
        estimator = Iter(MultiSURF(n_jobs=-1))
    elif algorithm == 'turf':
        estimator = TURF(MultiSURF(n_jobs=-1),num_scores_to_return=10)
    elif algorithm == 'vls_iter':
        estimator = Iter(VLS(MultiSURF(n_jobs=-1),random_state=random_state))
    elif algorithm == 'vls_turf':
        estimator = TURF(VLS(MultiSURF(n_jobs=-1),random_state=random_state),num_scores_to_return=10)
    else:
        raise Exception('Algorithm invalid')

    estimator.fit(features, labels)

    #Stop Job Timer
    job_time = time.time() - job_start_time

    # Write Scores
    score_dict = {}
    for feature_name,feature_score in zip(genetic_data.drop(class_label, axis=1).columns, estimator.feature_importances_):
        score_dict[feature_name] = feature_score

    sorted_features = sorted(score_dict, key=lambda x: score_dict[x], reverse=True)

    outfile = open(outfile,mode='w')
    outfile.write(algorithm+' Analysis Completed with REBATE\n')
    outfile.write('Run Time (sec): ' + str(job_time) + '\n')
    outfile.write('=== SCORES ===\n')
    n = 1
    for k in sorted_features:
        outfile.write(str(k) + '\t' + str(score_dict[k]) + '\t' + str(n) + '\n')
        n += 1
    outfile.close()

    print(algorithm+' '+datapath+' job complete')


if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]))