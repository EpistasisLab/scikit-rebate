# job_global_rba_rankings.py
import os
import argparse
import pandas as pd

def collect_rankings(root_dir, include_subdirs=None):
    """
    Collects all rankings_list.csv files from subdirectories under root_dir.
    Optionally filter which subdirectories to include.
    """
    all_dfs = []
    
    for sub in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        # If --include is specified, skip subdirs not in the list
        if include_subdirs and sub not in include_subdirs:
            continue

        if sub == "GAMETES_2.2_dev_peter_XOR": # for XOR, will only use the xor-2 and xor-3 configurations (excluding 4 and 5-way)
            # get path to xor-2 and xor-3 rankings_list.csv only (rankings_path1 & rankings_path2), turn it into df1 and df2, then combine them into df
            rankings_path_xor2 = os.path.join(sub_path, "xor_2", "a_100", "s_1600", "xor_2_a_100s_1600_EDM-1", "Results", "rankings_list.csv")
            rankings_path_xor3 = os.path.join(sub_path, "xor_3", "a_100", "s_1600", "xor_3_a_100s_1600_EDM-1", "Results", "rankings_list.csv")

            if not os.path.exists(rankings_path_xor2) or not os.path.exists(rankings_path_xor3):
                print(f"[WARN] rankings_list.csv not found in {sub_path} for xor-2 or xor-3, skipping.")
                continue

            try:
                df_xor2 = pd.read_csv(rankings_path_xor2, comment='#')  # ignore comment line with title
                df_xor2['subdir'] = "xor_2"

                df_xor3 = pd.read_csv(rankings_path_xor3, comment='#')  # ignore comment line with title
                df_xor3['subdir'] = "xor_3"

                df = pd.concat([df_xor2, df_xor3], ignore_index=True) # combining xor-2 and xor-3 into one df
            except Exception as e:
                print(f"[ERROR] Could not read {rankings_path_xor2} or {rankings_path_xor3}: {e}")
                continue
        else:
            rankings_path = os.path.join(sub_path, 'rankings_list.csv')

            if not os.path.exists(rankings_path):
                print(f"[WARN] rankings_list.csv not found in {sub_path}, skipping.")
                continue

            try:
                df = pd.read_csv(rankings_path, comment='#')  # ignore comment line with title
                df['subdir'] = sub
            except Exception as e:
                print(f"[ERROR] Could not read {rankings_path}: {e}")
                continue

        if "a_1000/" in rankings_path or "1-feature_1000_" in rankings_path:
            df['Rank'] = 1 + (df['Rank'] - 1) * 99 / 999
        elif "a_10000/" in rankings_path or "1-feature_10000_" in rankings_path:
            df['Rank'] = 1 + (df['Rank'] - 1) * 99 / 9999
        elif "a_20000/" in rankings_path:
            df['Rank'] = 1 + (df['Rank'] - 1) * 99 / 19999
        elif "a_50000/" in rankings_path:
            df['Rank'] = 1 + (df['Rank'] - 1) * 99 / 49999
        elif "a_100000/" in rankings_path or "1-feature_100000_" in rankings_path:
            df['Rank'] = 1 + (df['Rank'] - 1) * 99 / 99999

        all_dfs.append(df)
        print(f"[INFO] Loaded rankings_list.csv from {sub}")

    if not all_dfs:
        print("[ERROR] No valid rankings_list.csv files found.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def compute_global_summary(all_rankings_df):
    """
    Computes global mean and median rank per RBA.
    """
    summary = (
        all_rankings_df.groupby('RBA')['Rank']
        .agg(['mean', 'median'])
        .reset_index()
        .rename(columns={'mean': 'Mean', 'median': 'Median'})
    )
    summary.sort_values(by=['Mean', 'Median'], inplace=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate global rankings from multiple subdirectories.")
    parser.add_argument("root_dir", help="Top-level directory containing dataset subdirectories.")
    parser.add_argument("--include", nargs="+", default=None,
                        help="Optional list of subdirectories to include (default: all).")

    args = parser.parse_args()
    root_dir = args.root_dir

    # short names -> actual subdirectory names
    SHORT_TO_FULL_SUBDIR = {
        "maineff": "GAMETES_2.2_dev_peter_mainEff_Datasets_Loc_1_Qnt_2_Pop_100000",
        "core2wayepistasis": "GAMETES_2.2_dev_peter_core2wayEpistasis_Datasets_Loc_2_Qnt_2_Pop_100000",
        "xor": "GAMETES_2.2_dev_peter_XOR",
        "maineff_2": "GAMETES_2.2_dev_peter_mainEff_additive_2_Datasets_2Het_Loc_1_Qnt_2_Pop_100000",
        "maineff_4": "GAMETES_2.2_dev_peter_mainEff_additive_4_Datasets_2Het_Loc_1_Qnt_2_Pop_100000",
        "2wayepi": "GAMETES_2.2_dev_peter_2wayEpiHeterogeneity_Datasets_2Het_Loc_2_Qnt_2_Pop_100000",
        "epi_order": "GAMETES_2.2_dev_peter_epi_order_Datasets_Loc_3_Qnt_2_Pop_100000",
    }

    user_include = args.include

    if user_include:
        # map the user-specified short names to actual subdirectory names
        include_subdirs = []
        for short_name in user_include:
            # full_name = SHORT_TO_FULL_SUBDIR.get(short_name)
            full_name = SHORT_TO_FULL_SUBDIR.get(short_name.lower())
            if full_name:
                include_subdirs.append(full_name)
            else:
                print(f"[WARN] No mapping found for '{short_name}', skipping.")
    else:
        include_subdirs = None  # default: include all subdirectories
    
    print(f"[INFO] Subdirectories to include: {include_subdirs}")

    # NEW: going back up to data directory from AbsVal_Benchmark_Data to then enter Sanity_Check_Data
    parent_dir = os.path.dirname(root_dir)
    largerfeature_2way_dir = os.path.join(
        parent_dir,
        "Sanity_Check_Data",
        "benchmark-data",
        "Simulated_Benchmark_Archive",
        "GAMETES_2.2_dev_peter_2wayEpiFeatures_Datasets_Loc_2_Qnt_2_Pop_100000"
    )
    include_subdirs_largerfeature_2way = ["a_1000", "a_10000", "a_20000", "a_50000", "a_100000"]

    # now adding larger feature set (main effect) directory
    largerfeature_mainEff_dir = os.path.join(
        parent_dir,
        "mainEff_largerfeatures_data"
    )

    combined_df = collect_rankings(root_dir, include_subdirs)
    if combined_df is None:
        return
        
    # getting rankings from a_1000, a_10000, a_20000, a_50000, a_100000 (subdirs of largerfeature_2way_dir)
    largerfeature_2way_df = collect_rankings(largerfeature_2way_dir, include_subdirs_largerfeature_2way)

    largerfeature_mainEff_df = collect_rankings(largerfeature_mainEff_dir) # rankings for larger feature mainEff data

    # all rankings from all tested datasets with >= 100 features
    final_combined_df = pd.concat([combined_df, largerfeature_2way_df, largerfeature_mainEff_df], ignore_index=True)

    # all univariate effect data
    univariate_df = final_combined_df[
        final_combined_df['subdir'].isin(["GAMETES_2.2_dev_peter_mainEff_Datasets_Loc_1_Qnt_2_Pop_100000", # 1-Feature Main Effect
                                          "GAMETES_2.2_dev_peter_mainEff_additive_2_Datasets_2Het_Loc_1_Qnt_2_Pop_100000", # 2-Feature Additive Effect
                                          "GAMETES_2.2_dev_peter_mainEff_additive_4_Datasets_2Het_Loc_1_Qnt_2_Pop_100000", # 4-Feature Additive Effect
                                          "1-feature_1000_Feature_H_0.4_MAF_0.2_EDM-2", # 1,000 total features, main effect
                                          "1-feature_10000_Feature_H_0.4_MAF_0.2_EDM-2", # 10,000 total features, main effect
                                          "1-feature_100000_Feature_H_0.4_MAF_0.2_EDM-2" # 100,000 total features, main effect
                                          ])
    ].reset_index(drop=True)

    # all 2-way interaction data
    twoway_df = final_combined_df[
        final_combined_df['subdir'].isin(["GAMETES_2.2_dev_peter_core2wayEpistasis_Datasets_Loc_2_Qnt_2_Pop_100000", # 2-way Pure Epistasis
                                          "xor_2", # 2-way XOR
                                          "GAMETES_2.2_dev_peter_2wayEpiHeterogeneity_Datasets_2Het_Loc_2_Qnt_2_Pop_100000", # 2-way Epi Heterogeneity
                                          "a_1000", # 1,000 total features, 2-way interaction effect
                                          "a_10000", # 10,000 total features, 2-way interaction effect
                                          "a_20000", # 20,000 total features, 2-way interaction effect
                                          "a_50000", # 50,000 total features, 2-way interaction effect
                                          "a_100000" # 100,000 total features, 2-way interaction effect
                                          ])
    ].reset_index(drop=True)

    # all 3-way interaction data
    threeway_df = final_combined_df[
        final_combined_df['subdir'].isin(["GAMETES_2.2_dev_peter_epi_order_Datasets_Loc_3_Qnt_2_Pop_100000", # 3-way Pure Epistasis
                                          "xor_3" # 3-way XOR
                                          ])
    ].reset_index(drop=True)

    # mean and median metrics for each algorithm within each effect type (univariate, 2-way, 3-way)
    univariate_summary_df = compute_global_summary(univariate_df)
    twoway_summary_df = compute_global_summary(twoway_df)
    threeway_summary_df = compute_global_summary(threeway_df)

    # *** Creating global metrics: 2-way only, univariate + 2-way, univariate + 2-way + 3-way

    # creating a combined dataframe with univariate and 2-way metrics
    merged_uni_twoway_df = univariate_summary_df.merge(
        twoway_summary_df,
        on='RBA',
        suffixes=('_uni', '_two')
    )

    # creating the final weighted (univariate, 2-way) global metrics
    univariate_twoway_final_df = pd.DataFrame({
        'RBA': merged_uni_twoway_df['RBA'],
        'Avg_Mean': (
            (merged_uni_twoway_df['Mean_uni'] + merged_uni_twoway_df['Mean_two']) / 2
        ),
        'Avg_Median': (
            (merged_uni_twoway_df['Median_uni'] + merged_uni_twoway_df['Median_two']) / 2
        )
    })
    
    # sorting this dataframe
    univariate_twoway_final_df = (
        univariate_twoway_final_df
        .sort_values(by=['Avg_Mean', 'Avg_Median'], ascending=[True, True])
        .reset_index(drop=True)
    )

    # creating a combined dataframe with univariate, 2-way, and 3-way metrics
    merged_uni_twoway_threeway_df = merged_uni_twoway_df.merge(
        threeway_summary_df,
        on='RBA'
    )
    # for naming clarity
    merged_uni_twoway_threeway_df = merged_uni_twoway_threeway_df.rename(columns={
        'Mean': 'Mean_three',
        'Median': 'Median_three'
    })

    # creating the final weighted (univariate, 2-way, 3-way) global metrics
    univariate_twoway_threeway_final_df = pd.DataFrame({
        'RBA': merged_uni_twoway_threeway_df['RBA'],
        'Avg_Mean': (
            (merged_uni_twoway_threeway_df['Mean_uni'] +
            merged_uni_twoway_threeway_df['Mean_two'] +
            merged_uni_twoway_threeway_df['Mean_three']) / 3
        ),
        'Avg_Median': (
            (merged_uni_twoway_threeway_df['Median_uni'] +
            merged_uni_twoway_threeway_df['Median_two'] +
            merged_uni_twoway_threeway_df['Median_three']) / 3
        )
    })
    # univariate_twoway_threeway_final_df = pd.DataFrame({
    #     'RBA': merged_uni_twoway_threeway_df['RBA'],
    #     'Weighted_Mean': (
    #         merged_uni_twoway_threeway_df['Mean_uni'] * 0.4 +
    #         merged_uni_twoway_threeway_df['Mean_two'] * 0.4 +
    #         merged_uni_twoway_threeway_df['Mean_three'] * 0.2
    #     ),
    #     'Weighted_Median': (
    #         merged_uni_twoway_threeway_df['Median_uni'] * 0.4 +
    #         merged_uni_twoway_threeway_df['Median_two'] * 0.4 +
    #         merged_uni_twoway_threeway_df['Median_three'] * 0.2
    #     )
    # })
    # sorting this dataframe
    univariate_twoway_threeway_final_df = (
        univariate_twoway_threeway_final_df
        .sort_values(by=['Avg_Mean', 'Avg_Median'], ascending=[True, True])
        .reset_index(drop=True)
    )

    # --- Save final .csv's with metrics for (2-way only), (univariate + 2-way), (univariate + 2-way + 3-way) ---
    twoway_path = os.path.join(parent_dir, 'global_rba_rankings_2wayonly.csv')
    univariate_twoway_path = os.path.join(parent_dir, 'global_rba_rankings_uni2way.csv')
    univariate_twoway_threeway_path = os.path.join(parent_dir, 'global_rba_rankings_uni2way3way.csv')

    twoway_summary_df.to_csv(twoway_path, index=False)
    univariate_twoway_final_df.to_csv(univariate_twoway_path, index=False)
    univariate_twoway_threeway_final_df.to_csv(univariate_twoway_threeway_path, index=False)

    print(f"[INFO] Saved (2-way only) rankings to {twoway_path}")
    print(f"[INFO] Saved (univariate + 2-way) rankings to {univariate_twoway_path}")
    print(f"[INFO] Saved (univariate + 2-way + 3-way) rankings to {univariate_twoway_threeway_path}")
        


if __name__ == "__main__":
    main()