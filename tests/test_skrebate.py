import pandas as pd
import numpy as np
import pytest
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TURF, SWRFstar, SWRF, MultiSWRFstar, MultiSWRF, MultiSWRFDBstar, MultiSWRFDB, MuRelief
import warnings

warnings.filterwarnings('ignore')
np.random.seed(3249083)

# Load datasets once for all tests
# CI runs from repo root by default so path goes straight into data dir
# ** only selecting 200 instances from each dataset to speed up CI
datasets = {
    "binary": pd.read_csv('data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv').sample(n=200, random_state=3249083),
    "cont_endpoint": pd.read_csv('data/GAMETES_Epistasis_2-Way_continuous_endpoint_a_20s_1600her_0.4__maf_0.2_EDM-2_01.csv').sample(n=200, random_state=3249083),
    "mixed": pd.read_csv('data/GAMETES_Epistasis_2-Way_mixed_attribute_a_20s_1600her_0.4__maf_0.2_EDM-2_01.csv').sample(n=200, random_state=3249083),
    "missing": pd.read_csv('data/GAMETES_Epistasis_2-Way_missing_values_0.1_a_20s_1600her_0.4__maf_0.2_EDM-2_01.csv').sample(n=200, random_state=3249083),
    "multiclass": pd.read_csv('data/3Class_Datasets_Loc_2_01.csv').sample(n=200, random_state=3249083)
}

# Parametrize over selectors
selectors = [
    # Existing selectors
    (ReliefF, {'n_features_to_select': 5, 'n_neighbors': 1}), 
    (ReliefF, {'n_features_to_select': 5, 'n_neighbors': 10}),
    (SURF, {'n_features_to_select': 5}),
    (SURFstar, {'n_features_to_select': 5}),
    (MultiSURF, {'n_features_to_select': 5}),
    (MultiSURFstar, {'n_features_to_select': 5}),
    # (TURF, {'relief_object': ReliefF(n_features_to_select=5, n_neighbors=10), 'pct': 0.5, 'num_scores_to_return': 5}),
    # New algorithms
    (SWRFstar, {'n_features_to_select': 5}),
    (SWRF, {'n_features_to_select': 5}),
    (MultiSWRFstar, {'n_features_to_select': 5}),
    (MultiSWRF, {'n_features_to_select': 5}),
    (MultiSWRFDBstar, {'n_features_to_select': 5}),
    (MultiSWRFDB, {'n_features_to_select': 5}),
    (MuRelief, {'n_features_to_select': 5, 'n_neighbors': 1}), 
    (MuRelief, {'n_features_to_select': 5, 'n_neighbors': 10}),
]

@pytest.mark.parametrize("dataset_name,genetic_data", datasets.items())
@pytest.mark.parametrize("cls,kwargs", selectors)
def test_selector_on_dataset(cls, kwargs, dataset_name, genetic_data):
    features = genetic_data.drop('class', axis=1).values
    labels = genetic_data['class'].values
    
    selector = cls(**kwargs)
    selector.fit(features, labels)
    
    # Basic sanity checks after fitting
    assert hasattr(selector, "top_features_")
    assert hasattr(selector, "feature_importances_")
    assert len(selector.top_features_) == features.shape[1]
    assert len(selector.feature_importances_) == features.shape[1]
    assert np.isfinite(selector.feature_importances_).all()