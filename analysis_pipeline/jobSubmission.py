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
    features, labels = genetic_data.drop(class_label, axis=1).values, genetic_data[class_label].values

    #Customize hyperparameter values for the dataset
    num_scores_to_return = int(features[1]/2)         #Num scores to be nonzero at the end of TURF
    size_feature_subset = features[1]/2               #Size of VLS subset

    # Train
    if algorithm == 'multisurf':
        estimator = MultiSURF()
    elif algorithm == 'vls':
        estimator = VLS(MultiSURF(),random_state=random_state,size_feature_subset=size_feature_subset)
    elif algorithm == 'iter':
        estimator = Iter(MultiSURF())
    elif algorithm == 'turf':
        estimator = TURF(MultiSURF(),num_scores_to_return=num_scores_to_return)
    elif algorithm == 'vls_iter':
        estimator = Iter(VLS(MultiSURF(),random_state=random_state,size_feature_subset=size_feature_subset))
    elif algorithm == 'vls_turf':
        estimator = TURF(VLS(MultiSURF(),random_state=random_state,size_feature_subset=size_feature_subset),num_scores_to_return=num_scores_to_return)
    elif algorithm == 'multisurf_abs':
        estimator = MultiSURF(rank_absolute=True)
    elif algorithm == 'vls_abs':
        estimator = VLS(MultiSURF(rank_absolute=True),random_state=random_state,size_feature_subset=size_feature_subset)
    elif algorithm == 'iter_abs':
        estimator = Iter(MultiSURF(rank_absolute=True))
    elif algorithm == 'turf_abs':
        estimator = TURF(MultiSURF(rank_absolute=True),num_scores_to_return=num_scores_to_return)
    elif algorithm == 'vls_iter_abs':
        estimator = Iter(VLS(MultiSURF(rank_absolute=True),random_state=random_state,size_feature_subset=size_feature_subset))
    elif algorithm == 'vls_turf_abs':
        estimator = TURF(VLS(MultiSURF(rank_absolute=True),random_state=random_state,size_feature_subset=size_feature_subset),num_scores_to_return=num_scores_to_return)
    else:
        raise Exception('Algorithm invalid')

    estimator.fit(features, labels)

    #Stop Job Timer
    job_time = time.time() - job_start_time

    # Write Scores
    outfile = open(outfile,mode='w')
    outfile.write(algorithm+' Analysis Completed with REBATE\n')
    outfile.write('Run Time (sec): ' + str(job_time) + '\n')
    outfile.write('=== SCORES ===\n')

    feature_names = genetic_data.drop(class_label, axis=1).columns
    for feature_index in estimator.top_features_:
        outfile.write(str(feature_names[feature_index]) + '\t' + str(estimator.feature_importances_[feature_index]) + '\n')
    outfile.close()

    print(algorithm+' '+datapath+' job complete')


if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5])