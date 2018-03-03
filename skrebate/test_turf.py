# coding: utf-8
import pandas as pd
from turf import *
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
clf = make_pipeline(TuRF(core_algorithm="MultiSURF"),  RandomForestClassifier(n_estimators=100))

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
clf.fit(features, labels)
print(np.mean(cross_val_score(clf, features, labels)))
