import sys
import warnings
import numpy as np
import pandas as pd
import timeit

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar


warnings.filterwarnings('ignore')

np.random.seed(3249083)

genetic_data = pd.read_csv(
    'data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz',
     sep='\t',
     compression='gzip'
)
genetic_data = genetic_data.sample(frac=0.25)

# a larger dataset to be used for performance benchmarking sweeps
larger_data = pd.read_csv('data/test_model_Models.txt_EDM-1_1.txt', sep='\t')
larger_data.rename(columns={'Class': 'class'}, inplace=True)

# We're deliberately leaving all these unused datasets in here to make it
# easier to extend testing to more cases in the future.
genetic_data_cont_endpoint = pd.read_csv(
    'data/GAMETES_Epistasis_2-Way_continuous_endpoint_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz',
    sep='\t',
    compression='gzip'
)
genetic_data_cont_endpoint.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_cont_endpoint = genetic_data_cont_endpoint.sample(frac=0.25)

genetic_data_mixed_attributes = pd.read_csv(
    'data/GAMETES_Epistasis_2-Way_mixed_attribute_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz',
    sep='\t',
    compression='gzip'
)
genetic_data_mixed_attributes.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_mixed_attributes = genetic_data_mixed_attributes.sample(frac=0.25)

genetic_data_missing_values = pd.read_csv(
    'data/GAMETES_Epistasis_2-Way_missing_values_0.1_a_20s_1600her_0.4__maf_0.2_EDM-2_01.tsv.gz',
    sep='\t',
    compression='gzip'
)
genetic_data_missing_values.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_missing_values = genetic_data_missing_values.sample(frac=0.25)

genetic_data_multiclass = pd.read_csv(
    'data/3Class_Datasets_Loc_2_01.txt',
    sep='\t'
)
genetic_data_multiclass.rename(columns={'Class': 'class'}, inplace=True)
genetic_data_multiclass = genetic_data_multiclass.sample(frac=0.25)


features = genetic_data.drop('class', axis=1).values
labels = genetic_data['class'].values
headers = list(genetic_data.drop("class", axis=1))

larger_features = larger_data.drop('class', axis=1).values
larger_labels = larger_data['class'].values
larger_headers = list(larger_data.drop('class', axis=1))

features_cont_endpoint = genetic_data_cont_endpoint.drop(
    'class',
    axis=1
).values
labels_cont_endpoint = genetic_data_cont_endpoint['class'].values
headers_cont_endpoint = list(genetic_data_cont_endpoint.drop("class", axis=1))

features_mixed_attributes = genetic_data_mixed_attributes.drop(
    'class',
    axis=1
).values
labels_mixed_attributes = genetic_data_mixed_attributes['class'].values
headers_mixed_attributes = list(
    genetic_data_mixed_attributes.drop("class", axis=1)
)

features_missing_values = genetic_data_missing_values.drop(
    'class',
    axis=1
).values
labels_missing_values = genetic_data_missing_values['class'].values
headers_missing_values = list(
    genetic_data_missing_values.drop("class", axis=1)
)

features_multiclass, labels_multiclass = genetic_data_multiclass.drop(
    'class', axis=1).values, genetic_data_multiclass['class'].values
headers_multiclass = list(genetic_data_multiclass.drop("class", axis=1))


# Basic tests for performance of the algorithms in parallel and serial modes
def test_relieff():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): ReliefF Serial
    """
    np.random.seed(49082)

    alg = ReliefF(n_features_to_select=2, n_neighbors=10)
    alg.fit(features, labels)


def test_relieff_parallel():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): ReliefF parallel
    """
    np.random.seed(49082)

    alg = ReliefF(n_features_to_select=2, n_neighbors=10, n_jobs=-1)
    alg.fit(features, labels)

def test_relieff_parallel_larger_data():
    """
    Runtime: Data (Binary Endpoint, Discrete Features):
    ReliefF parallel larger data
    """
    np.random.seed(49082)

    alg = ReliefF(n_features_to_select=2, n_neighbors=10, n_jobs=-1)
    alg.fit(larger_features, larger_labels)

def test_relieffpercent():
    """
    Runtime: Data (Binary Endpoint, Discrete Features):
    ReliefF with % neighbors
    """
    np.random.seed(49082)

    alg = ReliefF(n_features_to_select=2, n_neighbors=0.1)
    alg.fit(features, labels)


def test_surf():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): SURF serial
    """
    np.random.seed(240932)

    alg = SURF(n_features_to_select=2)
    alg.fit(features, labels)


def test_surf_parallel():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): SURF parallel
    """
    np.random.seed(240932)

    alg = SURF(n_features_to_select=2, n_jobs=-1)
    alg.fit(features, labels)


def test_surfstar():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): SURF* serial
    """
    np.random.seed(9238745)

    alg = SURFstar(n_features_to_select=2)
    alg.fit(features, labels)


def test_surfstar_parallel():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): SURF* parallel
    """
    np.random.seed(9238745)

    alg = SURFstar(n_features_to_select=2, n_jobs=-1)
    alg.fit(features, labels)


def test_multisurfstar():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): MultiSURF* serial
    """
    np.random.seed(320931)

    alg = MultiSURFstar(n_features_to_select=2)
    alg.fit(features, labels)


def test_multisurfstar_parallel():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): MultiSURF* parallel
    """
    np.random.seed(320931)

    alg = MultiSURFstar(n_features_to_select=2, n_jobs=-1)
    alg.fit(features, labels)


def test_multisurf():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): MultiSURF serial
    """
    np.random.seed(320931)

    alg = MultiSURF(n_features_to_select=2)
    alg.fit(features, labels)


def test_multisurf_parallel():
    """
    Runtime: Data (Binary Endpoint, Discrete Features): MultiSURF parallel
    """
    np.random.seed(320931)

    alg = MultiSURF(n_features_to_select=2, n_jobs=-1)
    alg.fit(features, labels)


def run_parameter_sweep_binary_discrete():
    """
    Run a parameter sweep across several combinations of different row and col
    sizes for our test dataset. Saves a pandas df to disk that we can use
    for visualizations to confirm performance characteristics.

    To utilize, run `python performance_tests.py sweep`
    """
    row_sizes = [200, 400, 800, 1600, 3200]
    attr_sizes = [25, 50, 100, 200, 400]

    param_sweep_data = pd.DataFrame(columns=attr_sizes, index=row_sizes)

    for row_size in row_sizes:
        for attr_size in attr_sizes:
            # import pdb; pdb.set_trace()
            # sample down to the desired number of rows
            frac_to_sample = row_size / larger_data.shape[0]
            data_sample = larger_data.sample(frac=frac_to_sample)

            # sample down to the desired number of attr_size
            # len(cols) - attr_size - 1, the 1 is for the class column
            data_sample = data_sample.iloc[
                :,
                (len(data_sample.columns)-attr_size-1):
            ]

            features, labels = (
                data_sample.drop('class', axis=1).values,
                data_sample['class'].values
            )
            headers = list(data_sample.drop("class", axis=1))

            np.random.seed(49082)
            def run_algorithm():
                alg = ReliefF(
                    n_features_to_select=2,
                    n_neighbors=10,
                    n_jobs=-1
                )
                alg.fit(features, labels)

            timing = timeit.repeat(run_algorithm, number=1, repeat=5)

            param_sweep_data.loc[row_size,attr_size] = np.mean(timing[1:])

            print('%s rows, %s attributes: %s seconds' % (
                row_size,
                attr_size,
                np.mean(timing[1:]))
            )

    param_sweep_data.to_csv('param_sweep_data.csv')

test_cases = [
    test_relieff,
    test_relieff_parallel,
    test_relieff_parallel_larger_data,
    test_relieffpercent,
    # test_surf,
    # test_surf_parallel,
    # test_surfstar,
    # test_surfstar_parallel,
    # test_multisurfstar,
    # test_multisurfstar_parallel,
    # test_multisurf,
    # test_multisurf_parallel
]

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == 'sweep':
        run_parameter_sweep_binary_discrete()
    else:
        timing_df = pd.DataFrame(columns=['test_case', 'mean', 'std'])

        for test_case in test_cases:
            timing = timeit.repeat(test_case, number=1, repeat=5)
            # ignore the first test to avoid high initial overhead
            # to compile numba functions with small datasets
            timing = timing[1:]
            print(test_case.__name__, np.mean(timing), np.std(timing))
            d = {
                'test_case' : test_case.__name__,
                'mean' : np.mean(timing),
                'std' : np.std(timing)
            }
            timing_df = timing_df.append(d, ignore_index = True)

        print(timing_df)

        timing_df.to_csv('timing_benchmarks.csv')
