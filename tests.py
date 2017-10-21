from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

np.random.seed(3249083)

genetic_data = pd.read_csv('data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz', sep='\t', compression='gzip')
genetic_data = genetic_data.sample(frac=0.25)

genetic_data_cont_endpoint = pd.read_csv('data/GAMETES_Epistasis_2-Way_continuous_endpoint_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz', sep='\t', compression='gzip')
genetic_data_cont_endpoint.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_cont_endpoint = genetic_data_cont_endpoint.sample(frac=0.25)

genetic_data_mixed_attributes = pd.read_csv('data/GAMETES_Epistasis_2-Way_mixed_attribute_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz', sep='\t', compression='gzip')
genetic_data_mixed_attributes.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_mixed_attributes = genetic_data_mixed_attributes.sample(frac=0.25)

genetic_data_missing_values = pd.read_csv('data/GAMETES_Epistasis_2-Way_missing_values_0.1_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz', sep='\t', compression='gzip')
genetic_data_missing_values.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_missing_values = genetic_data_missing_values.sample(frac=0.25)

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
features_cont_endpoint, labels_cont_endpoint = genetic_data_cont_endpoint.drop('class', axis=1).values, genetic_data_cont_endpoint['class'].values
features_mixed_attributes, labels_mixed_attributes = genetic_data_mixed_attributes.drop('class', axis=1).values, genetic_data_mixed_attributes['class'].values
features_missing_values, labels_missing_values = genetic_data_missing_values.drop('class', axis=1).values, genetic_data_missing_values['class'].values

def test_relieff_init():
    """Ensure that the ReliefF constructor stores custom values correctly"""
    clf = ReliefF(n_features_to_select=7,
                  n_neighbors=500,
                  discrete_threshold=20,
                  verbose=True,
                  n_jobs=3)

    assert clf.n_features_to_select == 7
    assert clf.n_neighbors == 500
    assert clf.discrete_threshold == 20
    assert clf.verbose == True
    assert clf.n_jobs == 3

def test_surf_init():
    """Ensure that the SURF, SURF*, and MultiSURF constructors store custom values correctly"""
    clf = SURF(n_features_to_select=7,
               discrete_threshold=20,
               verbose=True,
               n_jobs=3)

    assert clf.n_features_to_select == 7
    assert clf.discrete_threshold == 20
    assert clf.verbose == True
    assert clf.n_jobs == 3

### Parallelization tests

def test_relieff_pipeline():
    """Ensure that ReliefF works in a sklearn pipeline when it is parallelized"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7

def test_relieff_pipeline_parallel():
    """Ensure that ReliefF works in a sklearn pipeline where cross_val_score is parallelized"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7

def test_relieffpercent_pipeline():
    """Ensure that ReliefF with % neighbors works in a sklearn pipeline when it is parallelized"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=0.1, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7

def test_relieffpercent_pipeline_parallel():
    """Ensure that ReliefF with % neighbors works in a sklearn pipeline where cross_val_score is parallelized"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=0.1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7

def test_surf_pipeline():
    """Ensure that SURF works in a sklearn pipeline when it is parallelized"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7

def test_surf_pipeline_parallel():
    """Ensure that SURF works in a sklearn pipeline where cross_val_score is parallelized"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7

def test_surfstar_pipeline():
    """Ensure that SURF* works in a sklearn pipeline when it is parallelized"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7

def test_surfstar_pipeline_parallel():
    """Ensure that SURF* works in a sklearn pipeline where cross_val_score is parallelized"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7

def test_multisurf_pipeline():
    """Ensure that MultiSURF works in a sklearn pipeline when it is parallelized"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7

def test_multisurf_pipeline_parallel():
    """Ensure that MultiSURF works in a sklearn pipeline where cross_val_score is parallelized"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7

def test_multisurfstar_pipeline():
    """Ensure that MultiSURF* works in a sklearn pipeline when it is parallelized"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7

def test_multisurfstar_pipeline_parallel():
    """Ensure that MultiSURF* works in a sklearn pipeline where cross_val_score is parallelized"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7

### Test algorithms with data that has continuous endpoints

def test_relieff_pipeline_cont_endpoint():
    """Ensure that ReliefF works in a sklearn pipeline with continuous endpoint data"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100, n_jobs=-1),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint, labels_cont_endpoint, cv=3))) < 0.5

def test_relieff_pipeline_cont_endpoint():
    """Ensure that ReliefF with % neighbors works in a sklearn pipeline with continuous endpoint data"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=0.1, n_jobs=-1),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint, labels_cont_endpoint, cv=3))) < 0.5

def test_surf_pipeline_cont_endpoint():
    """Ensure that SURF works in a sklearn pipeline with continuous endpoint data"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2, n_jobs=-1),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint, labels_cont_endpoint, cv=3))) < 0.5

def test_surfstar_pipeline_cont_endpoint():
    """Ensure that SURF* works in a sklearn pipeline with continuous endpoint data"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2, n_jobs=-1),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint, labels_cont_endpoint, cv=3))) < 0.5

def test_multisurf_pipeline_cont_endpoint():
    """Ensure that MultiSURF works in a sklearn pipeline with continuous endpoint data"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2, n_jobs=-1),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint, labels_cont_endpoint, cv=3))) < 0.5

def test_multisurfstar_pipeline_cont_endpoint():
    """Ensure that MultiSURF* works in a sklearn pipeline with continuous endpoint data"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2, n_jobs=-1),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint, labels_cont_endpoint, cv=3))) < 0.5

### Test algorithms with data that has mixed attributes

def test_relieff_pipeline_mixed_attributes():
    """Ensure that ReliefF works in a sklearn pipeline with mixed attributes"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes, labels_mixed_attributes, cv=3)) > 0.7

def test_relieffpercent_pipeline_mixed_attributes():
    """Ensure that ReliefF with % neighbors works in a sklearn pipeline with mixed attributes"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=0.1, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes, labels_mixed_attributes, cv=3)) > 0.7

def test_surf_pipeline_mixed_attributes():
    """Ensure that SURF works in a sklearn pipeline with mixed attributes"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes, labels_mixed_attributes, cv=3)) > 0.7

def test_surfstar_pipeline_mixed_attributes():
    """Ensure that SURF* works in a sklearn pipeline with mixed attributes"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes, labels_mixed_attributes, cv=3)) > 0.7

def test_multisurf_pipeline_mixed_attributes():
    """Ensure that MultiSURF works in a sklearn pipeline with mixed attributes"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes, labels_mixed_attributes, cv=3)) > 0.7

def test_multisurfstar_pipeline_mixed_attributes():
    """Ensure that MultiSURF* works in a sklearn pipeline with mixed attributes"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes, labels_mixed_attributes, cv=3)) > 0.7

### Test algorithms with data that has missing values

def test_relieff_pipeline_missing_values():
    """Ensure that ReliefF works in a sklearn pipeline with missing values"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100, n_jobs=-1),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values, labels_missing_values, cv=3)) > 0.7

def test_relieffpercent_pipeline_missing_values():
    """Ensure that ReliefF with % neighbors works in a sklearn pipeline with missing values"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=0.1, n_jobs=-1),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values, labels_missing_values, cv=3)) > 0.7

def test_surf_pipeline_missing_values():
    """Ensure that SURF works in a sklearn pipeline with missing values"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2, n_jobs=-1),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values, labels_missing_values, cv=3)) > 0.7

def test_surfstar_pipeline_missing_values():
    """Ensure that SURF* works in a sklearn pipeline with missing values"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2, n_jobs=-1),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values, labels_missing_values, cv=3)) > 0.7

def test_multisurf_pipeline_missing_values():
    """Ensure that MultiSURF works in a sklearn pipeline with missing values"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2, n_jobs=-1),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values, labels_missing_values, cv=3)) > 0.7

def test_multisurfstar_pipeline_missing_values():
    """Ensure that MultiSURF* works in a sklearn pipeline with missing values"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2, n_jobs=-1),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values, labels_missing_values, cv=3)) > 0.7
