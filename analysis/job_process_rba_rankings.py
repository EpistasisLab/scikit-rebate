# job_process_rba_rankings.py
import os
import sys
import argparse
import pandas as pd
import numpy as np

def collect_rba_rankings(root_dir):
    print(f"Searching for Results folders under: {root_dir}")
    all_rankings_df = pd.DataFrame()

    # dict to create cleaner RBA names in mean/median rank tables
    rba_descriptive_names = {
        'RandomShuffle': 'Random Shuffle',
        'MutualInfo': 'Mutual Info',
        'ReliefF10': 'ReliefF 10NN',
        'ReliefF100': 'ReliefF 100NN',
        'SURF': 'SURF',
        'SURFstar': 'SURF*',
        'MultiSURF': 'MultiSURF',
        'MultiSURFstar': 'MultiSURF*',
        'SWRF': 'SWRF',
        'SWRFstar': 'SWRF*',
        'MultiSWRF': 'MultiSWRF',
        'MultiSWRFstar': 'MultiSWRF*',
        'MultiSWRFDB': 'MultiSWRFDB',
        'MultiSWRFDBstar': 'MultiSWRFDB*',
        'MuRelief10': 'Mu-Relief 10N',
        'MuRelief100': 'Mu-Relief 100N',
    }

    for subdir, dirs, _ in os.walk(root_dir):
        subgroup_rankings_df = pd.DataFrame()
        # Only process Results folders that are within these directories
        if os.path.basename(subdir) == "Results" and any(x in subdir for x in ["a_100", "a_1000", "a_10000", "a_100000", "mainEff_largerfeatures_data", "a_50000", "a_20000", "a_2000", "a_5000", "a_8000"]):
            for rba in os.listdir(subdir):
                if rba not in rba_descriptive_names:
                    continue
                rba_path = os.path.join(subdir, rba)
                if not os.path.isdir(rba_path):
                    continue

                for file in os.listdir(rba_path):
                    if not file.endswith('.txt'):
                        continue
                    file_path = os.path.join(rba_path, file)

                    if "Shuffle" in rba:
                        try:
                            df = pd.read_csv(file_path, sep='\t', usecols=['Feature'])
                        except Exception as e:
                            print(f"Skipping {file_path} (error: {e})")
                            continue

                        df['Rank'] = df.index + 1
                        df['Feature_Importance'] = np.nan
                        df['Normalized_Feature_Importance'] = np.nan
                    else:
                        try:
                            df = pd.read_csv(file_path, sep='\t', usecols=['Feature', 'Feature_Importance'])
                        except Exception as e:
                            print(f"Skipping {file_path} (error: {e})")
                            continue

                        # Sort by descending feature importance and assign ranks
                        df.sort_values(by='Feature_Importance', ascending=False, inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        df['Rank'] = df.index + 1
                        # Normalize feature importance between 0 and 1
                        df['Normalized_Feature_Importance'] = (df['Feature_Importance'] - df['Feature_Importance'].min()) / \
                                                                (df['Feature_Importance'].max() - df['Feature_Importance'].min())

                    # Only keep true predictive features
                    predictive_df = df[df['Feature'].str.startswith('M')][['Feature', 'Feature_Importance', 'Normalized_Feature_Importance', 'Rank']]
                    predictive_df['RBA'] = rba_descriptive_names[rba]

                    # for subgroup ranking (ex. mainEff, her=0.2, EDM-1); i.e. group of 30 replicate dataset files
                    subgroup_rankings_df = pd.concat([subgroup_rankings_df, predictive_df], ignore_index=True)
                    # for whole dataset group ranking (ex. mainEff)
                    all_rankings_df = pd.concat([all_rankings_df, predictive_df], ignore_index=True)

            compute_summary_stats(subgroup_rankings_df, subdir)

    compute_summary_stats(all_rankings_df, root_dir)


def compute_summary_stats(rankings_df, save_dir):
    if rankings_df.empty:
        print("Valid ranking data not found.")
        sys.exit(0)

    summary = (
        rankings_df.groupby('RBA')['Rank']
        .agg(['mean', 'median'])
        .reset_index()
        .rename(columns={'mean': 'Mean', 'median': 'Median'})
    )

    summary.sort_values(by=['Mean', 'Median'], inplace=True)

    # Get title from basename of save_dir
    if os.path.basename(os.path.normpath(save_dir)) == "Results":
        title = os.path.basename(os.path.dirname(save_dir))
    else:
        title = os.path.basename(os.path.normpath(save_dir))

    summary_path = os.path.join(save_dir, 'rba_rankings.csv')

    rankings_list_path = os.path.join(save_dir, 'rankings_list.csv')

    # Write summary CSV with title
    with open(summary_path, 'w') as f:
        f.write(f"# {title}\n")
        summary.to_csv(f, index=False)
    print(f"Saved summary CSV: {summary_path}")

    # Write detailed rankings CSV with title
    with open(rankings_list_path, 'w') as f:
        f.write(f"# {title}\n")
        rankings_df[['Feature', 'Feature_Importance', 'Normalized_Feature_Importance', 'Rank', 'RBA']].to_csv(f, index=False)
    print(f"Saved detailed rankings list: {rankings_list_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute global mean and median ranks for RBAs.")
    parser.add_argument("root_dir", help="Path to the dataset group directory (e.g., path/to/mainEff).")
    args = parser.parse_args()

    collect_rba_rankings(args.root_dir)

    print("Completed processing for:", args.root_dir)


if __name__ == "__main__":
    main()