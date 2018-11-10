import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate.turf import TuRF
from skrebate.vlsrelief import VLSRelief
from skrebate.newalgo import NewAlgo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
headers = list(genetic_data.drop("class", axis=1))
fs = NewAlgo(core_algorithm="ReliefF", n_features_to_select=2,pct=0.5,verbose=True)
fs.fit(features, labels, headers)
for feature_name, feature_score in zip(genetic_data.drop('class', axis=1).columns, fs.feature_importances_):
    print(feature_name, '\t', feature_score)