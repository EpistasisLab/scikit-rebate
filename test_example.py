import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF, SURF, SURFstar, MultiSURF
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data_link = ('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
            'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz')

genetic_data = pd.read_csv(data_link, sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# ReliefF

clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100, n_jobs=-1),
                    RandomForestClassifier(n_estimators=100))

print('ReliefF',np.mean(cross_val_score(clf, features, labels)))


# SURF

clf = make_pipeline(SURF(n_features_to_select=2, n_jobs=-1),
                    RandomForestClassifier(n_estimators=100))

"""

print('SURF',np.mean(cross_val_score(clf, features, labels)))

# SURF*

clf = make_pipeline(SURFstar(n_features_to_select=2, n_jobs=-1),
                    RandomForestClassifier(n_estimators=100))

print('SURF*',np.mean(cross_val_score(clf, features, labels)))

# MultiSURF

clf = make_pipeline(MultiSURF(n_features_to_select=2, n_jobs=-1),
                    RandomForestClassifier(n_estimators=100))

print('MultiSURF',np.mean(cross_val_score(clf, features, labels)))

# TURF

clf = make_pipeline(RFE(ReliefF(n_jobs=-1), n_features_to_select=2),
                    RandomForestClassifier(n_estimators=100))

print('TURF',np.mean(cross_val_score(clf, features, labels)))
"""
