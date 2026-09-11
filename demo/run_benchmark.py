import os
import sys
import pandas as pd
import numpy as np
import argparse
import time

from multiprocessing import Pool, cpu_count

from sklearn.feature_selection import mutual_info_classif

# Get the absolute path to the directory containing the package
package_path = os.path.abspath(os.path.join("", ".."))

# Add the package path to sys.path
sys.path.insert(0, package_path)
from skrebate import SURFstar, MultiSURFstar, SWRFstar

def find_txt_groups(root_dir):
    # Grab all the folders that contain 30 txt files and exclude the file name that contains 'Results'
    txt_groups = []
    
    for root, dirs, files in os.walk(root_dir):
        txt_files = [f for f in files if f.endswith('.txt') and 'Results' not in f]

        files_count = len(txt_files)
        
        # if files_count == 30 and 'Results' not in root:
        #     txt_groups.append(root)
        # elif files_count != 0:
        #     print(f"{root} got {len(txt_files)} files")
        if 'Results' not in root:
            txt_groups.append(root)
        elif files_count != 0:
            print(f"{root} got {len(txt_files)} files")
    
    return txt_groups


def single_file_job(args):
    # Run all the methods on a single data file
    exp_dir, txt_file, result_dir, methods = args

    try:
        print(f"\tProcessing with data file {txt_file}...")
        if txt_file.endswith('.txt'):   
            data = pd.read_csv(os.path.join(exp_dir, txt_file), sep='\t')
        elif txt_file.endswith('.csv'):
            data = pd.read_csv(os.path.join(exp_dir, txt_file))
        else:
            raise ValueError(f"Unsupported file format: {txt_file}")
        features, labels = data.drop('class', axis=1).values, data['class'].values

        for method_name, (method, params) in methods.items():
            if method_name == "RandomShuffle":
                importances = np.random.permutation(len(data.drop('class', axis=1).columns))
            elif method_name == "MutualInformation":
                importances = mutual_info_classif(features, labels)
            else:        
                fs = method(**params)
                fs.fit(features, labels)
                importances = fs.feature_importances_

            results = pd.DataFrame({
                'Feature': data.drop('class', axis=1).columns,
                'Feature_Importance': importances
            })

            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            results.to_csv(os.path.join(result_dir, exp_dir, method_name, f"{base_name}_Results.txt"), index=False, sep='\t')
        return f"File {txt_file} processed successfully."
    except Exception as e:
        raise e  # Uncomment this line to raise the exception after logging the error
        return f"Error processing {txt_file}: {e}"


def main(data_dir, result_dir, methods, num_cpus, dry_run):
    # Find all the folder that contains 30 data files
    exp_dirs = find_txt_groups(data_dir)
    print(f"Found {len(exp_dirs)} exps")

    if dry_run:
        print("[Dry Run] The files in the following folders will be processed:")
        for exp_dir in exp_dirs:
            print(f"{exp_dir}")
        return

    # Generate the configuration for all the jobs
    args = []
    for exp_dir in exp_dirs:

        for method_name, _ in methods.items():
            os.makedirs(os.path.join(result_dir, exp_dir, method_name), exist_ok=True)
        
        txt_files = [f for f in os.listdir(exp_dir) if (f.endswith('.txt') or f.endswith('.csv'))]

        args += [(exp_dir, txt_file, result_dir, methods) for txt_file in txt_files]

    start_time = time.time()

    # results = []
    # for arg in args:
    #     results.append(single_file_job(arg))  # Test a single file job to ensure everything is working

    print(f"Parallel processing on {len(args)} tasks...")
    with Pool(processes=num_cpus) as pool:
        results = pool.map(single_file_job, args)

    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel processing for feature importance computation.")
    # parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .txt files.")
    # parser.add_argument("--result_dir", type=str, default='.', help="Path to the directory for saving results.")
    parser.add_argument("--num_cpus", type=int, default=None, help="Number of CPU cores to use for multiprocessing.")
    parser.add_argument("-n", "--dry_run", action="store_true", help="If set, perform a dry run without actual computation.")
    args = parser.parse_args()

    if args.num_cpus is None:
        num_cpus = cpu_count()
    else:
        num_cpus = args.num_cpus
    print(f"Using cpu number: {num_cpus}")


    ############Path and method setting############
    data_dir = "../data/"
    result_dir = 'Results'

    methods = {
        "RandomShuffle": (None, None),
        "MutualInformation": (None, None),
        "SURFstar": (SURFstar, {'n_features_to_select': 5}),
        "MultiSURFstar": (MultiSURFstar, {'n_features_to_select': 5}),
        "SWRFstar": (SWRFstar, {'n_features_to_select': 5}),
    }
    ############Path and method setting############

    main(data_dir, result_dir, methods, args.num_cpus, args.dry_run)

