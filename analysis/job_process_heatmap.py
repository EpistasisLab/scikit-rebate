import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def process_results_dir(results_dir, prefix=""):
    print(f"Processing: {results_dir}")
    
    all_rankings_df = pd.DataFrame()
    total_features_per_rba = {}

    for rba in os.listdir(results_dir):
        rba_path = os.path.join(results_dir, rba)
        if os.path.isdir(rba_path):
            method_feature_counts = []
            for file in os.listdir(rba_path):
                if file.endswith('.txt'):
                    file_path = os.path.join(rba_path, file)
                    parts = file.split('_')
                    dataset_id = '_'.join(parts[-2].split('_')[:2])

                    if rba == "RandomShuffle":
                        df = pd.read_csv(file_path, sep='\t', usecols=['Feature'])
                        df['Rank'] = df.index + 1
                    else:
                        column_to_use = 'ABS_Feature_Importance' if "ABS" in rba else 'Feature_Importance'
                        df = pd.read_csv(file_path, sep='\t', usecols=['Feature', column_to_use])
                        df.sort_values(by=column_to_use, ascending=False, inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        df['Rank'] = df.index + 1

                    predictive_df = df[df['Feature'].str.startswith('M')][['Feature', 'Rank']]
                    predictive_df['RBA'] = rba
                    predictive_df['Dataset'] = dataset_id

                    all_rankings_df = pd.concat([all_rankings_df, predictive_df], ignore_index=True)
                    method_feature_counts.append(df['Feature'].nunique())

            total_features_per_rba[rba] = max(method_feature_counts)

    rankings_path = os.path.join(results_dir, 'consolidated_rankings.csv')
    all_rankings_df.to_csv(rankings_path, index=False)

    N = next(iter(total_features_per_rba.values()))
    lowest_ranks = all_rankings_df.groupby(['RBA', 'Dataset'])['Rank'].max().reset_index()
    percentages = {rba: [0] * N for rba in lowest_ranks['RBA'].unique()}
    for rba in percentages:
        rba_data = lowest_ranks[lowest_ranks['RBA'] == rba]
        for pos in range(1, N + 1):
            count_higher = rba_data[rba_data['Rank'] <= pos].shape[0]
            percentages[rba][pos - 1] = (count_higher / rba_data.shape[0]) * 100

    percentages_df = pd.DataFrame(percentages, index=range(1, N + 1))
    perc_path = os.path.join(results_dir, 'percentages_df.csv')
    percentages_df.to_csv(perc_path, index_label='Ranking Position')

    # Plot heatmap
    # custom_cmap = sns.color_palette('Oranges', n_colors=1000)[:800] + sns.color_palette('Blues', n_colors=1000)[800:]
    # ** Heatmap tweaked so that 100% is a distinct purple shade and 0% is a distinct white
    colors = sns.color_palette('Oranges', n_colors=1000)[:800] + sns.color_palette('Blues', n_colors=1000)[800:]
    colors = np.array(colors) # converting to mutable array
    colors[0] = [1.0, 1.0, 1.0] # replacing the lowest value color with white (for 0% on the heatmap)
    colors[-1] = [0.5, 0.0, 0.5] # replacing the highest value color with purple (for 100% on the heatmap)
    custom_cmap = ListedColormap(colors)

    # Define your preferred order of the RBAs as a list
    rba_order = [
        'RandomShuffle',
        'MutualInfo',
        'ReliefF10',
        'ReliefF100',
        'MuRelief10',
        'MuRelief100',
        'SURF',
        'MultiSURF',
        'SWRF',
        'MultiSWRF',
        'MultiSWRFDB',
        'SURFstar',
        'MultiSURFstar',
        'SWRFstar',
        'MultiSWRFstar',
        'MultiSWRFDBstar',
    ]

    # Define a mapping from your RBA order to new descriptive names
    rba_descriptive_names = {
        'RandomShuffle': 'Random Shuffle',
        'MutualInfo': 'Mutual Info',
        'ReliefF10': 'ReliefF 10NN',
        'ReliefF100': 'ReliefF 100NN',
        'MuRelief10': 'Mu-Relief 10N',
        'MuRelief100': 'Mu-Relief 100N',
        'SURF': 'SURF',
        'MultiSURF': 'MultiSURF',
        'SWRF': 'SWRF',
        'MultiSWRF': 'MultiSWRF',
        'MultiSWRFDB': 'MultiSWRFDB',
        'SURFstar': 'SURF*',
        'MultiSURFstar': 'MultiSURF*',
        'SWRFstar': 'SWRF*',
        'MultiSWRFstar': 'MultiSWRF*',
        'MultiSWRFDBstar': 'MultiSWRFDB*',
    }

    n_pred = all_rankings_df['Feature'].nunique()
    percentages_df = percentages_df.iloc[n_pred-1:]
    percentages_df_transposed = percentages_df.T
    percentages_df_ordered = percentages_df_transposed.loc[[r for r in rba_order if r in percentages_df_transposed.index]]
    xtick_labels = ['Optimal', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
    xtick_positions = np.linspace(0, percentages_df_ordered.shape[1] - 0.13, num=len(xtick_labels))

    plt.figure(figsize=(12, 7))
    print(percentages_df_ordered)
    heatmap = sns.heatmap(percentages_df_ordered, annot=False, fmt=".1f", cmap=custom_cmap, vmin=0, vmax=100, cbar_kws={'label': 'Power (Frequency of Success)'}) # explicit 0% and 100% for min and max values that heatmap colorscheme corresponds to
    # heatmap = sns.heatmap(percentages_df_ordered, annot=False, fmt=".1f", cmap=custom_cmap, vmin=0, vmax=100, cbar=False) # if no color bar legend is desired
    for i in range(percentages_df_ordered.shape[0] - 1):
        heatmap.axhline(i + 1, color='black', linewidth=1.5)

    for _, spine in heatmap.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    cbar = heatmap.collections[0].colorbar
    cbar.outline.set_linewidth(1.5)
    cbar.outline.set_edgecolor("black")
    # Increase label font size
    cbar.set_label('Power (Frequency of Success)', fontsize=16)
    cbar.ax.yaxis.set_label_coords(3.0, 0.5) # explicitly placing the cbar label closer to the cbar
    # Increase tick label size
    cbar.ax.tick_params(labelsize=14)
    # adding clear labeling of 0% power as white and 100% power as purple
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([
        '0 (white)',
        '20',
        '40',
        '60',
        '80',
        '100 (purple)'
    ])

    dataset_id = os.path.basename(os.path.dirname(results_dir))
    # heatmap.set_title(prefix + dataset_id, fontsize=16)
    heatmap.set_xlabel('Predictive features in top % of ranked features', fontsize=16)
    # heatmap.set_ylabel('Method', fontsize=16)
    heatmap.set_xticks(xtick_positions)
    heatmap.set_xticklabels(xtick_labels, rotation=0, fontsize=16)
    new_ytick_labels = [rba_descriptive_names[rba] for rba in percentages_df_ordered.index]
    heatmap.set_yticklabels(new_ytick_labels, fontsize=16)
    # to remove xticks altogether:
    # heatmap.set_xticks([])

    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    heatmap.collections[0].set_rasterized(True) # rasterizing the heatmap to improve rendering of PDF with high feature counts

    save_path = os.path.join(results_dir, prefix + dataset_id + '.pdf')
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight') # high dpi for higher resolution/quality rasterized heatmap
    plt.close()

    print(f"Completed processing for: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to the Results folder.")
    parser.add_argument("--prefix", default="", help="Prefix to add to output heatmap filenames.")
    args = parser.parse_args()

    process_results_dir(args.results_dir, prefix=args.prefix)
