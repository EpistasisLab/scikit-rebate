
"""
scikit-rebate was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Pete Schmitt (pschmitt@upenn.edu)
    - Ryan J. Urbanowicz (ryanurb@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - and many more generous open source contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

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

genetic_data = pd.read_csv(
    'data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz', sep='\t', compression='gzip')
genetic_data = genetic_data.sample(frac=0.25)

genetic_data_cont_endpoint = pd.read_csv(
    'data/GAMETES_Epistasis_2-Way_continuous_endpoint_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz', sep='\t', compression='gzip')
genetic_data_cont_endpoint.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_cont_endpoint = genetic_data_cont_endpoint.sample(frac=0.25)

genetic_data_mixed_attributes = pd.read_csv(
    'data/GAMETES_Epistasis_2-Way_mixed_attribute_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz', sep='\t', compression='gzip')
genetic_data_mixed_attributes.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_mixed_attributes = genetic_data_mixed_attributes.sample(frac=0.25)

genetic_data_missing_values = pd.read_csv(
    'data/GAMETES_Epistasis_2-Way_missing_values_0.1_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz', sep='\t', compression='gzip')
genetic_data_missing_values.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_missing_values = genetic_data_missing_values.sample(frac=0.25)

genetic_data_multiclass = pd.read_csv('data/3Class_Datasets_Loc_2_01.txt', sep='\t')
genetic_data_multiclass.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_multiclass = genetic_data_multiclass.sample(frac=0.25)


features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
headers = list(genetic_data.drop("class", axis=1))

features_cont_endpoint, labels_cont_endpoint = genetic_data_cont_endpoint.drop(
    'class', axis=1).values, genetic_data_cont_endpoint['class'].values
headers_cont_endpoint = list(genetic_data_cont_endpoint.drop("class", axis=1))

features_mixed_attributes, labels_mixed_attributes = genetic_data_mixed_attributes.drop(
    'class', axis=1).values, genetic_data_mixed_attributes['class'].values
headers_mixed_attributes = list(genetic_data_mixed_attributes.drop("class", axis=1))

features_missing_values, labels_missing_values = genetic_data_missing_values.drop(
    'class', axis=1).values, genetic_data_missing_values['class'].values
headers_missing_values = list(genetic_data_missing_values.drop("class", axis=1))

features_multiclass, labels_multiclass = genetic_data_multiclass.drop(
    'class', axis=1).values, genetic_data_multiclass['class'].values
headers_multiclass = list(genetic_data_multiclass.drop("class", axis=1))

# Initialization tests--------------------------------------------------------------------------------
def test_relieff_init():
    """Check: ReliefF constructor stores custom values correctly"""
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
    """Check: SURF, SURF*, and MultiSURF constructors store custom values correctly"""
    clf = SURF(n_features_to_select=7,
               discrete_threshold=20,
               verbose=True,
               n_jobs=3)

    assert clf.n_features_to_select == 7
    assert clf.discrete_threshold == 20
    assert clf.verbose == True
    assert clf.n_jobs == 3

# Basic Parallelization Tests and Core binary data and discrete feature data testing (Focus on ReliefF only for efficiency)------------------------------------------------------------
def test_relieff_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): ReliefF works in a sklearn pipeline"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=10),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7


def test_relieff_pipeline_parallel():
    """Check: Data (Binary Endpoint, Discrete Features): ReliefF works in a sklearn pipeline when ReliefF is parallelized"""
    # Note that the rebate algorithm cannot be parallelized with both the random forest and the cross validation all at once.  If the rebate algorithm is parallelized, the cross-validation scoring cannot be.
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=10, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7


def test_relieffpercent_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): ReliefF with % neighbors works in a sklearn pipeline"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=0.1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7


def test_surf_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): SURF works in a sklearn pipeline"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7


def test_surf_pipeline_parallel():
    """Check: Data (Binary Endpoint, Discrete Features): SURF works in a sklearn pipeline when SURF is parallelized"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7


def test_surfstar_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): SURF* works in a sklearn pipelined"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7


def test_surfstar_pipeline_parallel():
    """Check: Data (Binary Endpoint, Discrete Features): SURF* works in a sklearn pipeline when SURF* is parallelized"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7


def test_multisurfstar_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): MultiSURF* works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7


def test_multisurfstar_pipeline_parallel():
    """Check: Data (Binary Endpoint, Discrete Features): MultiSURF* works in a sklearn pipeline when MultiSURF* is parallelized"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7


def test_multisurf_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): MultiSURF works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.7


def test_multisurf_pipeline_parallel():
    """Check: Data (Binary Endpoint, Discrete Features): MultiSURF works in a sklearn pipeline when MultiSURF is parallelized"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2, n_jobs=-1),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.7


def test_turf_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): TuRF with ReliefF works in a sklearn pipeline"""
    np.random.seed(49082)

    # clf = make_pipeline(TuRF(core_algorithm="ReliefF", n_features_to_select=2, pct=0.5, n_neighbors=100),
    #                     RandomForestClassifier(n_estimators=100, n_jobs=-1))
    #
    # assert np.mean(cross_val_score(clf, features, labels, fit_params={
    #                'turf__headers': headers}, cv=3, n_jobs=-1)) > 0.7


def test_turf_pipeline_parallel():
    """Check: Data (Binary Endpoint, Discrete Features): TuRF with ReliefF works in a sklearn pipeline when TuRF is parallelized"""
    np.random.seed(49082)

    # clf = make_pipeline(TuRF(core_algorithm="ReliefF", n_features_to_select=2, pct=0.5, n_neighbors=100, n_jobs=-1),
    #                     RandomForestClassifier(n_estimators=100, n_jobs=-1))
    #
    # assert np.mean(cross_val_score(clf, features, labels, fit_params={
    #                'turf__headers': headers}, cv=3)) > 0.7


def test_vlsrelief_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): VLSRelief with ReliefF works in a sklearn pipeline"""
    np.random.seed(49082)

    # clf = make_pipeline(VLSRelief(core_algorithm="ReliefF", n_features_to_select=2, n_neighbors=100),
    #                     RandomForestClassifier(n_estimators=100, n_jobs=-1))
    #
    # assert np.mean(cross_val_score(clf, features, labels, fit_params={
    #                'vlsrelief__headers': headers}, cv=3, n_jobs=-1)) > 0.7


def test_vlsrelief_pipeline_parallel():
    """Check: Data (Binary Endpoint, Discrete Features): VLSRelief with ReliefF works in a sklearn pipeline when VLSRelief is parallelized"""
    np.random.seed(49082)

    # clf = make_pipeline(VLSRelief(core_algorithm="ReliefF", n_features_to_select=2, n_neighbors=100, n_jobs=-1),
    #                     RandomForestClassifier(n_estimators=100, n_jobs=-1))
    #
    # assert np.mean(cross_val_score(clf, features, labels, fit_params={
    #                'vlsrelief__headers': headers}, cv=3)) > 0.7


def test_iterrelief_pipeline():
    """Check: Data (Binary Endpoint, Discrete Features): IterRelief with ReliefF works in a sklearn pipeline"""
    np.random.seed(49082)

    # clf = make_pipeline(IterRelief(core_algorithm="ReliefF", n_features_to_select=2, n_neighbors=100),
    #                     RandomForestClassifier(n_estimators=100, n_jobs=-1))
    #
    # assert np.mean(cross_val_score(clf, features, labels, cv=3, n_jobs=-1)) > 0.5


def test_iterrelief_pipeline_parallel():
    """Check: Data (Binary Endpoint, Discrete Features): IterRelief with ReliefF works in a sklearn pipeline when VLSRelief is parallelized"""
    np.random.seed(49082)

    # clf = make_pipeline(IterRelief(core_algorithm="ReliefF", n_features_to_select=2, n_neighbors=100, n_jobs=-1),
    #                     RandomForestClassifier(n_estimators=100, n_jobs=-1))
    #
    # assert np.mean(cross_val_score(clf, features, labels, cv=3)) > 0.5

# Test Multiclass Data ------------------------------------------------------------------------------------


def test_relieff_pipeline_multiclass():
    """Check: Data (Multiclass Endpoint): ReliefF works in a sklearn pipeline """
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=10),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_multiclass,
                                   labels_multiclass, cv=3, n_jobs=-1)) > 0.7


def test_surf_pipeline_multiclass():
    """Check: Data (Multiclass Endpoint): SURF works in a sklearn pipeline"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_multiclass,
                                   labels_multiclass, cv=3, n_jobs=-1)) > 0.7


def test_surfstar_pipeline_multiclass():
    """Check: Data (Multiclass Endpoint): SURF* works in a sklearn pipeline"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_multiclass,
                                   labels_multiclass, cv=3, n_jobs=-1)) > 0.7


def test_multisurfstar_pipeline_multiclass():
    """Check: Data (Multiclass Endpoint): MultiSURF* works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_multiclass,
                                   labels_multiclass, cv=3, n_jobs=-1)) > 0.7


def test_multisurf_pipeline_multiclass():
    """Check: Data (Multiclass Endpoint): MultiSURF works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_multiclass,
                                   labels_multiclass, cv=3, n_jobs=-1)) > 0.7


# Test Continuous Endpoint Data ------------------------------------------------------------------------------------

def test_relieff_pipeline_cont_endpoint():
    """Check: Data (Continuous Endpoint): ReliefF works in a sklearn pipeline"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=10),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint,
                                       labels_cont_endpoint, cv=3, n_jobs=-1))) < 0.5


def test_surf_pipeline_cont_endpoint():
    """Check: Data (Continuous Endpoint): SURF works in a sklearn pipeline"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint,
                                       labels_cont_endpoint, cv=3, n_jobs=-1))) < 0.5


def test_surfstar_pipeline_cont_endpoint():
    """Check: Data (Continuous Endpoint): SURF* works in a sklearn pipeline"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint,
                                       labels_cont_endpoint, cv=3, n_jobs=-1))) < 0.5


def test_multisurfstar_pipeline_cont_endpoint():
    """Check: Data (Continuous Endpoint): MultiSURF* works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint,
                                       labels_cont_endpoint, cv=3, n_jobs=-1))) < 0.5


def test_multisurf_pipeline_cont_endpoint():
    """Check: Data (Continuous Endpoint): MultiSURF works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2),
                        RandomForestRegressor(n_estimators=100, n_jobs=-1))

    assert abs(np.mean(cross_val_score(clf, features_cont_endpoint,
                                       labels_cont_endpoint, cv=3, n_jobs=-1))) < 0.5

# Test Mixed Attribute Data ------------------------------------------------------------------------------------


def test_relieff_pipeline_mixed_attributes():
    """Check: Data (Mixed Attributes): ReliefF works in a sklearn pipeline"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=10),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes,
                                   labels_mixed_attributes, cv=3, n_jobs=-1)) > 0.7


def test_surf_pipeline_mixed_attributes():
    """Check: Data (Mixed Attributes): SURF works in a sklearn pipeline"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes,
                                   labels_mixed_attributes, cv=3, n_jobs=-1)) > 0.7


def test_surfstar_pipeline_mixed_attributes():
    """Check: Data (Mixed Attributes): SURF* works in a sklearn pipeline"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes,
                                   labels_mixed_attributes, cv=3, n_jobs=-1)) > 0.7


def test_multisurfstar_pipeline_mixed_attributes():
    """Check: Data (Mixed Attributes): MultiSURF* works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes,
                                   labels_mixed_attributes, cv=3, n_jobs=-1)) > 0.7


def test_multisurf_pipeline_mixed_attributes():
    """Check: Data (Mixed Attributes): MultiSURF works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_mixed_attributes,
                                   labels_mixed_attributes, cv=3, n_jobs=-1)) > 0.7

# Test Missing Value Data ------------------------------------------------------------------------------------


def test_relieff_pipeline_missing_values():
    """Check: Data (Missing Values): ReliefF works in a sklearn pipeline"""
    np.random.seed(49082)

    clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=10),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values,
                                   labels_missing_values, cv=3, n_jobs=-1)) > 0.7


def test_surf_pipeline_missing_values():
    """Check: Data (Missing Values): SURF works in a sklearn pipeline"""
    np.random.seed(240932)

    clf = make_pipeline(SURF(n_features_to_select=2),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values,
                                   labels_missing_values, cv=3, n_jobs=-1)) > 0.7


def test_surfstar_pipeline_missing_values():
    """Check: Data (Missing Values): SURF* works in a sklearn pipeline"""
    np.random.seed(9238745)

    clf = make_pipeline(SURFstar(n_features_to_select=2),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values,
                                   labels_missing_values, cv=3, n_jobs=-1)) > 0.7


def test_multisurfstar_pipeline_missing_values():
    """Check: Data (Missing Values): MultiSURF* works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURFstar(n_features_to_select=2),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values,
                                   labels_missing_values, cv=3, n_jobs=-1)) > 0.7


def test_multisurf_pipeline_missing_values():
    """Check: Data (Missing Values): MultiSURF works in a sklearn pipeline"""
    np.random.seed(320931)

    clf = make_pipeline(MultiSURF(n_features_to_select=2),
                        Imputer(),
                        RandomForestClassifier(n_estimators=100, n_jobs=-1))

    assert np.mean(cross_val_score(clf, features_missing_values,
                                   labels_missing_values, cv=3, n_jobs=-1)) > 0.7
