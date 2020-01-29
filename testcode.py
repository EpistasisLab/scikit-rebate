import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate.relieff import ReliefF
from skrebate.iterrelief import IterRelief

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split

genetic_data = pd.read_csv(
    'data/a_20s_1600her_0.4__maf_0.2_EDM-2_01.txt', sep='\t')
genetic_data = genetic_data.sample(frac=0.25)

features, labels = genetic_data.drop('Class', axis=1).values, genetic_data['Class'].values

clf = make_pipeline(IterRelief(core_algorithm = 'multisurf',n_features_to_select=2, n_neighbors=100),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))


#
# Make sure to compute the feature importance scores from only your training set
# X_train, X_test, y_train, y_test = train_test_split(features, labels)
#
# fs = IterRelief(core_algorithm = 'multisurf',n_features_to_select=2, n_neighbors=100)
# fs.fit(X_train, y_train)
#
# for feature_name, feature_score in zip(genetic_data.drop('Class', axis=1).columns,
#                                        fs.feature_importances_):
#     print(feature_name, '\t', feature_score)

#print(sum(genetic_data.loc[:,'M0P1'].isna()))

#print(genetic_data.loc[:,'M0P0'].isna().sum())
