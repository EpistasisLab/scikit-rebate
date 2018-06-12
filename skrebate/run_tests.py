import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
#from multisurf import MultiSURF
from relieff import ReliefF
from iterrelief import IterRelief
# from surf import SURF
# from surfstar import SURFstar
# from turf import TuRF
#from vlsrelief import VLSRelief
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# genetic_data = pd.read_csv(
#     'data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz', sep='\t', compression='gzip')
# genetic_data = genetic_data.sample(frac=0.25)

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
                           sep='\t', compression='gzip')

# genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
#                            'GAMETES_Epistasis_2-Way_continuous_endpoint_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz',
#                            sep='\t', compression='gzip')

print(genetic_data['class'].unique())

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
headers = list(genetic_data.drop("class", axis=1))

# clf = make_pipeline(VLSRelief(core_algorithm="MultiSURF", n_features_to_select=2, step=0.4, n_neighbors=100, n_jobs=-1),
#                     RandomForestClassifier(n_estimators=100, n_jobs=-1))

# clf = make_pipeline(VLSRelief(core_algorithm="ReliefF", n_features_to_select=2, step=0.4, n_neighbors=100, n_jobs=-1),
#                     RandomForestClassifier(n_estimators=100, n_jobs=-1))

# print(np.mean(cross_val_score(clf, features, labels, fit_params={
#     'vlsrelief__headers': headers}, cv=3)))


# x = VLSRelief(core_algorithm='ReliefF', num_feature_subset=3)
# # x = ReliefF()
# f = x.fit(features, labels, headers)
#
# print('scores=', f.features_scores_iter)
# print('features=', f.features_selected)
# print('head=', f.headers)
# print('idx=', f.headers_model)
# print('feat_score=', f.feature_importances_)
# print('top=', f.top_features_)
# print('h_top=', f.header_top_features_)
#
# print(f.feat_score)
#
# f2 = f.transform(f.X_mat)
# print(f2)
#
# print('X')
# print(f.X_mat[:2])

# print(x._class_type)
# print(x.mcmap)
# print(x.attr)
#
# # clf = make_pipeline(SURF(n_features_to_select=2),
# #                     RandomForestClassifier(n_estimators=100))
#

x = IterRelief('surf', weight_flag=2, n_features_to_select=2, n_neighbors=100)


f = x.fit(features, labels)
print(f.feature_importances_)

# clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
#                     RandomForestClassifier(n_estimators=100))

#
# clf = make_pipeline(SURFstar(n_features_to_select=2),
#                     RandomForestClassifier(n_estimators=100))
#
#
# print(np.mean(cross_val_score(clf, features, labels)))
#
#
# for feature_name, feature_score in zip(genetic_data.drop('class', axis=1).columns,
#                                        f.feature_importances_):
#     print(feature_name, '\t', feature_score)

# clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
#                     RandomForestClassifier(n_estimators=100))
#
# print(np.mean(cross_val_score(clf, features, labels)))
