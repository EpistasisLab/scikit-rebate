import os
import time
import sys
import argparse
import pandas as pd
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import hashlib
from functools import partial

package_path = os.path.abspath(os.path.join("", ".."))
sys.path.insert(0, package_path)
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, SWRFstar, SWRF, MultiSWRFstar, MultiSWRF, MultiSWRFDBstar, MultiSWRFDB, MuRelief

# NEW: added exist_ok=True
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def process_and_save_results(file_path, fs, method_name):
    df = pd.read_csv(file_path, sep='\t')
    
    X, y = df.drop('Class', axis=1).values, df['Class'].values

    # to keep track of runtime for large feature datasets
    start_time = time.time()
    try:
        fs.fit(X, y)
    except AttributeError:
        scores = fs(X, y)
        fs = type('MI', (), {'feature_importances_': scores})()
    end_time = time.time()

    temp_list = []
    for feature_name, feature_score in zip(df.drop('Class', axis=1).columns, fs.feature_importances_):
        temp_list.append([feature_name, feature_score])
    
    Results = pd.DataFrame(temp_list, columns=['Feature', 'Feature_Importance'])
    ABSResults = Results.copy()
    ABSResults['ABS_Feature_Importance'] = ABSResults['Feature_Importance'].abs()
    Results.sort_values(by='Feature_Importance', ascending=False, inplace=True)
    ABSResults.sort_values(by='ABS_Feature_Importance', ascending=False, inplace=True)

    # in case Mutual Info produces many FI scores tied at 0
    if method_name == "MutualInfo":
        # can be used for both regular and ABS, since MutualInfo doesn't produce negative scores
        zero_mask = Results['Feature_Importance'] == 0.0

        shuffled_zero_indices = (
            Results.loc[zero_mask]
            .sample(frac=1, random_state=42)
            .index
        )

        Results.loc[zero_mask] = Results.loc[shuffled_zero_indices].to_numpy()
        ABSResults.loc[zero_mask] = ABSResults.loc[shuffled_zero_indices].to_numpy()

    base_dir = os.path.dirname(file_path)
    results_dir = os.path.join(base_dir, "Results")
    method_dir = os.path.join(results_dir, method_name)
    abs_method_dir = os.path.join(results_dir, f"ABS_{method_name}")
    ensure_dir(method_dir)
    ensure_dir(abs_method_dir)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    Results.to_csv(os.path.join(method_dir, f"{base_name}_Results.txt"), index=False, sep='\t')
    ABSResults.to_csv(os.path.join(abs_method_dir, f"{base_name}_ABSResults.txt"), index=False, sep='\t')
    # Save runtime CSV
    runtime = end_time - start_time
    runtime_df = pd.DataFrame([{
        "Dataset": base_name,
        "Algorithm": method_name,
        "Runtime (sec)": round(runtime, 4),
        "Runtime (min)": round(runtime / 60, 4)
    }])
    runtime_df.to_csv(os.path.join(method_dir, f"{base_name}_runtime_postnanhandling.csv"), index=False)

    # ******** If I uncomment this function, I need to go back and uncomment logging lines within other files
    # if method_name in ["SWRFstar", "SWRF", "MultiSWRF", "MultiSWRFstar", "MultiSWRFDB", "MultiSWRFDBstar", "MultiSWRFDBlinear", "MultiSWRFDBlinearstar", "MultiSWRFDBexponential", "MultiSWRFDBexponentialstar", "MultiSWRFDBlinear3SD", "MultiSWRFDBlinear3SDstar", "MultiSWRFDBexponential3SD", "MultiSWRFDBexponential3SDstar", "SURF", "SURFstar", "MultiSURF", "MultiSURFstar", "MuRelief10", "MuRelief100"]:
    #     fs.plot_distance_weight_map(save_fig=os.path.join(method_dir, f"{base_name}_WeightPlot.png"), show_expected=True)
    #     # fs.plot_distance_weight_map(save_fig=os.path.join(method_dir, f"{base_name}_WeightPlot.png"), save_file=os.path.join(method_dir, f"{base_name}_stdweightlog.txt"), show_expected=True)

    print(f"Processed {file_path} with {method_name}. Results saved to {method_dir} and {abs_method_dir}.")

def process_random_shuffle(file_path):
    # Define the directory to store results
    results_dir = os.path.join(os.path.dirname(file_path), "Results", "RandomShuffle")
    ensure_dir(results_dir)

    # Read the file
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Shuffle feature column names (excluding 'Class' if present)
    if 'Class' in df.columns:
        columns_to_shuffle = sorted(df.drop('Class', axis=1).columns.tolist())
    else:
        columns_to_shuffle = sorted(df.columns.tolist())

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Take the last 2 characters of base_name (i.e. the file number)
    seed_str = base_name[-2:]  # example: "01"
    # Convert to deterministic integer seed
    file_seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)

    for i in range(40): # 40 random shuffles per replicate file (40 X 30 = 1200 random shuffles per configuration)
        seed = file_seed + i
        # creating a local RNG seeded from the file name
        rng = default_rng(seed)
        # shuffle columns deterministically for this file
        shuffled_columns = rng.permutation(columns_to_shuffle)

        shuffled_df = pd.DataFrame(shuffled_columns, columns=['Feature'])

        output_path = os.path.join(results_dir, f"{base_name}{i}_RandShuffle.txt")
        shuffled_df.to_csv(output_path, index=False, sep='\t')

def process_mutual_info(file_path):
    df = pd.read_csv(file_path, sep='\t')
    # counting the number of labels in the 'Class' column to determine whether this is a classification or regression problem:
    num_labels = df['Class'].nunique()
    if num_labels <= 10:
        fs = partial(mutual_info_classif, random_state=42, discrete_features=True) # setting random_state for mutual_info, and ensuring features are treated as discrete/categorical
    else:
        fs = partial(mutual_info_regression, random_state=42, discrete_features=True)
    process_and_save_results(file_path, fs, "MutualInfo")

def process_relieff10(file_path):
    fs = ReliefF(n_features_to_select=2,n_neighbors=10,n_jobs=1)
    process_and_save_results(file_path, fs, "ReliefF10")

def process_relieff100(file_path):
    fs = ReliefF(n_features_to_select=2,n_neighbors=100,n_jobs=16)
    process_and_save_results(file_path, fs, "ReliefF100")

def process_surf(file_path):
    fs = SURF(n_jobs=1)
    process_and_save_results(file_path, fs, "SURF")

def process_surfstar(file_path):
    fs = SURFstar(n_jobs=16)
    process_and_save_results(file_path, fs, "SURFstar")

def process_multisurf(file_path):
    fs = MultiSURF(n_jobs=16)
    process_and_save_results(file_path, fs, "MultiSURF")

def process_multisurfstar(file_path):
    fs = MultiSURFstar(n_jobs=16)
    process_and_save_results(file_path, fs, "MultiSURFstar")

def process_swrfstar(file_path):
    fs = SWRFstar(n_jobs=1)
    process_and_save_results(file_path, fs, "SWRFstar")

def process_swrf(file_path):
    fs = SWRF(n_jobs=16)
    process_and_save_results(file_path, fs, "SWRF")

def process_multiswrfstar(file_path):
    fs = MultiSWRFstar(n_jobs=16)
    process_and_save_results(file_path, fs, "MultiSWRFstar")

def process_multiswrf(file_path):
    fs = MultiSWRF(n_jobs=16)
    process_and_save_results(file_path, fs, "MultiSWRF")

def process_multiswrfdbstar(file_path):
    fs = MultiSWRFDBstar(n_jobs=16)
    process_and_save_results(file_path, fs, "MultiSWRFDBstar")

def process_multiswrfdb(file_path):
    fs = MultiSWRFDB(n_jobs=16)
    process_and_save_results(file_path, fs, "MultiSWRFDB")

def process_murelief10(file_path):
    fs = MuRelief(n_features_to_select=2,n_neighbors=10,n_jobs=1)
    process_and_save_results(file_path, fs, "MuRelief10")

def process_murelief100(file_path):
    fs = MuRelief(n_features_to_select=2,n_neighbors=100,n_jobs=1)
    process_and_save_results(file_path, fs, "MuRelief100")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, help='Algorithm to use')
    parser.add_argument('--input_file', required=True, help='Path to the input .txt file')
    args = parser.parse_args()

    alg_map = {
        'random': process_random_shuffle,
        'mutual_info': process_mutual_info,
        'relieff10': process_relieff10,
        'relieff100': process_relieff100,
        'surf': process_surf,
        'surfstar': process_surfstar,
        'multisurf': process_multisurf,
        'multisurfstar': process_multisurfstar,
        'swrfstar': process_swrfstar,
        'swrf': process_swrf,
        'multiswrfstar': process_multiswrfstar,
        'multiswrf': process_multiswrf,
        'multiswrfdbstar': process_multiswrfdbstar,
        'multiswrfdb': process_multiswrfdb,
        'murelief10': process_murelief10,
        'murelief100': process_murelief100,
    }

    if args.algorithm not in alg_map:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    alg_map[args.algorithm](args.input_file)

if __name__ == "__main__":
    main()
