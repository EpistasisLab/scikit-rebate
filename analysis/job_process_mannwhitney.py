# job_process_mannwhitney.py
import os
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def permutation_test(x, y, U_obs, n_permutations=10000, seed=42):
    np.random.seed(seed)
    combined = np.concatenate([x, y])
    count = 0
    n_x = len(x)
    expected_U = len(x) * len(y) / 2
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        new_x = combined[:n_x]
        new_y = combined[n_x:]
        # U_perm = U score
        U_perm = mannwhitneyu(new_x, new_y, alternative="two-sided", method="asymptotic").statistic

        if abs(U_perm - expected_U) >= abs(U_obs - expected_U):
            count += 1
    return count / n_permutations

def process_dir(dir_path, column='rank', exclude_patterns=None):
    exclude_patterns = exclude_patterns or []
    df = pd.read_csv(os.path.join(dir_path, 'rankings_list.csv'), comment='#')

    # Normalize column names
    col_to_use = 'Rank' if column == 'rank' else 'Normalized_Feature_Importance'

    # Group by RBA
    rba_groups = {rba: g[col_to_use].values for rba, g in df.groupby('RBA') 
                  if not any(p.lower() in rba.lower() for p in exclude_patterns)}

    results = []

    for rba1, rba2 in combinations(rba_groups.keys(), 2):
        x = rba_groups[rba1]
        y = rba_groups[rba2]
        
        mannwhitney_res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        row = {
            'RBA1': rba1,
            'RBA2': rba2,
            'mannwhitney_statistic': mannwhitney_res.statistic,
            'mannwhitney_pvalue': mannwhitney_res.pvalue
        }

        # Permutation test only for rank
        if column == 'rank':
            perm_p = permutation_test(x, y, mannwhitney_res.statistic)
            row['permutation_pvalue'] = perm_p

        results.append(row)

    results_df = pd.DataFrame(results)

    # Sorting
    if column == 'rank':
        # Benjamini-Hochberg
        results_df['mannwhitney_p_adj'] = multipletests(results_df['mannwhitney_pvalue'], method='fdr_bh')[1]
        results_df['permutation_p_adj'] = multipletests(results_df['permutation_pvalue'], method='fdr_bh')[1]
        results_df.sort_values(by=['mannwhitney_p_adj', 'permutation_p_adj'], ascending=True, inplace=True)
        output_file = os.path.join(dir_path, 'mannwhitney_ranks.csv')
    else:
        # Benjamini-Hochberg
        results_df['mannwhitney_p_adj'] = multipletests(results_df['mannwhitney_pvalue'], method='fdr_bh')[1]
        results_df.sort_values(by=['mannwhitney_p_adj'], ascending=True, inplace=True)
        # output_file = os.path.join(dir_path, 'mannwhitney_feature_importances.csv')
        output_file = os.path.join(dir_path, 'mannwhitney_normalized_feature_importances.csv')

    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", help="Directory containing rankings_list.csv")
    parser.add_argument("--column", choices=['rank', 'feature_importance'], default='rank', help="Column to test")
    parser.add_argument("--exclude", nargs='*', default=[], help="RBA name patterns to exclude (case insensitive)")
    args = parser.parse_args()

    process_dir(args.dir_path, column=args.column, exclude_patterns=args.exclude)