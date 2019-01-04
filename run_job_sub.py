
def main(dataset, outfile, algorithm):

    genetic_data = pd.read_csv(dataset, sep='\t', compression='gzip')
    tBefore = time.time()

    features, labels = genetic_data.drop('Class', axis=1).values, genetic_data['Class'].values
    headers = list(genetic_data.drop('Class', axis=1))
    if algorithm == "turf":
    	fs = TuRF(core_algorithm="MultiSURF", n_features_to_select=2,pct=0.5,verbose=True)
    	fs.fit(features, labels, headers)
    elif algorithm == "vls":
        
    	fs = VLSRelief(core_algorithm="MultiSURF", n_features_to_select=2,n_neighbors = 100,verbose=True)
    	fs.fit(features, labels, headers)
    elif algorithm == "iter":
    	fs = IterRelief(core_algorithm="MultiSURF", weight_flag=2,n_features_to_select=2,verbose=True)
    	fs.fit(features, labels)
    else:
    	print("invalid")
    print (features, labels, headers)
    
    scoreDict = {}
    for feature_name, feature_score in zip(genetic_data.drop('Class', axis=1).columns, fs.feature_importances_):
        scoreDict[feature_name] = feature_score

    sorted_names = sorted(scoreDict, key=lambda x: scoreDict[x], reverse=True)
    tAfter = (time.time() - tBefore)

    fh = open(outfile, 'w')
    fh.write(outfile)
    fh.write(algorithm + ' Analysis Completed with REBATE\n')
    fh.write('Run Time (sec): ' + str(tAfter) + '\n')
    fh.write('=== SCORES ===\n')
    n = 1
    for k in sorted_names:
        fh.write(str(k) + '\t' + str(scoreDict[k])  + '\t' + str(n) +'\n')
        n+=1
    fh.close()


if __name__=="__main__":
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from skrebate import MultiSURF
    from skrebate import TuRF
    from skrebate.vlsrelief import VLSRelief
    from skrebate.iterrelief import IterRelief

    
    import sys
    import os
    import time
    
    dataset = sys.argv[1]
    outfile = sys.argv[2]
    algorithm = sys.argv[3]
    print(dataset, outfile, algorithm)
    main(dataset, outfile, algorithm)