import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import time

# Helper to create the percentages_df (copied from old job_process_heatmap)
def compute_percentages(results_dir):
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

            total_features_per_rba[rba] = max(method_feature_counts) if method_feature_counts else 0

    N = next(iter(total_features_per_rba.values()))
    lowest_ranks = all_rankings_df.groupby(['RBA', 'Dataset'])['Rank'].max().reset_index()
    percentages = {rba: [0] * N for rba in lowest_ranks['RBA'].unique()}
    for rba in percentages:
        rba_data = lowest_ranks[lowest_ranks['RBA'] == rba]
        for pos in range(1, N + 1):
            count_higher = rba_data[rba_data['Rank'] <= pos].shape[0]
            percentages[rba][pos - 1] = (count_higher / rba_data.shape[0]) * 100 if rba_data.shape[0] > 0 else 0

    percentages_df = pd.DataFrame(percentages, index=range(1, N + 1))
    n_pred = all_rankings_df['Feature'].nunique()
    # print("N_pred:", n_pred, "\n")
    # pd.set_option('display.max_rows', None)
    # print(percentages_df["RandomShuffle"])
    # pd.reset_option('display.max_rows')  # optional: restore default
    return percentages_df.iloc[n_pred-1:]  # same trimming

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir", help="Directory containing multiple Results folders.")
    parser.add_argument("--prefix", default="", help="Prefix for unified PDF filename.")
    args = parser.parse_args()

    # custom colormap & ordering
    # custom_cmap = sns.color_palette('Oranges', n_colors=1000)[:800] + sns.color_palette('Blues', n_colors=1000)[800:]
    # ** Heatmap tweaked so that 100% is a distinct purple shade and 0% is a distinct white
    colors = sns.color_palette('Oranges', n_colors=1000)[:800] + sns.color_palette('Blues', n_colors=1000)[800:]
    colors = np.array(colors) # converting to mutable array
    colors[0] = [1.0, 1.0, 1.0] # replacing the lowest value color with white (for 0% on the heatmap)
    colors[-1] = [0.5, 0.0, 0.5] # replacing the highest value color with purple (for 100% on the heatmap)
    custom_cmap = ListedColormap(colors)

    # MAIN:
    rba_order = [
        'RandomShuffle','MutualInfo','ReliefF10','ReliefF100','MuRelief10','MuRelief100','SURF',
        'MultiSURF','SWRF','MultiSWRF','MultiSWRFDB','SURFstar','MultiSURFstar','SWRFstar','MultiSWRFstar','MultiSWRFDBstar'
    ]

    # Grab all Results folders (for 100 feature datasets)
    # * can later potentially add the dataset feature lengths you want for the heatmap as a parameter (ex. 100 for a_100 datasets)
    results_dirs = []
    for root, dirs, _ in os.walk(args.basedir):
        for d in dirs:
            if d == 'Results' and 'a_100' in root:
                results_dirs.append(os.path.join(root, d))

    is_core2wayEpistasis = ('core2wayEpistasis' in args.basedir)
    is_mainEff = ('mainEff_Datasets' in args.basedir)

    # Build a mapping of (n_instances, heritability, EDMtype) -> percentages_df
    pattern_n = re.compile(r"/s_(\d+)")
    pattern_h = re.compile(r"her_(\d+\.\d+)__maf")
    pattern_edm = re.compile(r"EDM-(\d+)")

    data_dict = {}
    for rd in results_dirs:
        print("Results directory:", rd)
        n_match = pattern_n.search(rd)
        h_match = pattern_h.search(rd)
        edm_match = pattern_edm.search(rd)
        if n_match and h_match and edm_match:
            n = int(n_match.group(1))
            h = float(h_match.group(1))
            edm = edm_match.group(1)  # '1' or '2'
            perc_df = compute_percentages(rd)
            perc_df_T = perc_df.T
            perc_df_ordered = perc_df_T.loc[[r for r in rba_order if r in perc_df_T.index]]
            data_dict[(n, h, edm)] = perc_df_ordered

    n_values = sorted({k[0] for k in data_dict.keys()})
    h_values = sorted({k[1] for k in data_dict.keys()})
    edm_values = sorted({k[2] for k in data_dict.keys()})
    print("Data dictionary:", data_dict, "\n")
    print("H values:", h_values, "\n")
    print("N values:", n_values, "\n")

    # --- INCREASING SPACING
    # only mainEff and core2wayEpistasis have both EDM-1 and EDM-2
    if is_core2wayEpistasis:
        total_rows = len(h_values) * 2
    else:
        total_rows = len(edm_values) # 2 rows for mainEff

    if is_mainEff:
        total_cols = len(h_values) # = 4
    else:
        total_cols = len(n_values)

    # Create custom height and width ratios
    # --> insert an extra gap between certain rows by slightly enlarging the space below
    height_ratios = []
    for i in range(total_rows):
        height_ratios.append(1)
        if is_core2wayEpistasis:
            if i % 2 == 1 and i != total_rows - 1:  # after every 2nd row (except last)
                # height_ratios.append(0.15)  # this adds vertical gap spacing
                height_ratios.append(0.05)
        else:
            if i != total_rows - 1:
                if is_mainEff:
                    height_ratios.append(0.10)

    # For columns, add an extra gap after every column
    width_ratios = []
    for j in range(total_cols):
        width_ratios.append(1)
        if j != total_cols - 1:  # after each column except last
            # width_ratios.append(0.10)  # horizontal gap spacing
            if is_core2wayEpistasis or is_mainEff:
                width_ratios.append(0.05)

    # Define figure and gridspec with custom spacing
    if is_core2wayEpistasis:
        fig = plt.figure(figsize=(4*len(n_values) * 1.15, 3*len(h_values)*2 * 1.1))
    elif is_mainEff:
        fig = plt.figure(figsize=(4*len(h_values) * 1.15, 4*len(edm_values) * 1.15))
    
    if is_core2wayEpistasis:
        spacing_value = 0.10
    elif is_mainEff:
        spacing_value = 0.05
    gs = gridspec.GridSpec(
        nrows=len(height_ratios),
        ncols=len(width_ratios),
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=spacing_value,  # fine-tune base spacing
        wspace=spacing_value
    )

    # Build axes array only in the actual plot cells (skip gap cells)
    axes = np.empty((total_rows, total_cols), dtype=object)
    row_ptr, col_ptr = 0, 0
    for i in range(len(height_ratios)):
        if is_core2wayEpistasis:
            if i % 3 == 2:  # skip gap row
                continue
        else:
            if i % 2 == 1:  # skip gap row
                continue
        col_ptr = 0
        for j in range(len(width_ratios)):
            if j % 2 == 1:  # skip gap column
                continue
            axes[row_ptr, col_ptr] = fig.add_subplot(gs[i, j])
            col_ptr += 1
        row_ptr += 1

    if len(h_values)*2 == 1 and len(n_values) == 1:
        axes = np.array([[axes]])  # ensure 2D

    # where to draw black lines separating RBA groups in each heatmap
    separators = [2, 6, 11]
    if is_core2wayEpistasis:
        # flipped so that y-axis is shown in ascending order from bottom to top
        for i, h in enumerate(sorted(h_values, reverse=True)):
            for j, n in enumerate(n_values):
                # top EDM-2
                ax_top = axes[i*2, j] if len(h_values)*2 > 1 else axes[0, j]
                ax_bot = axes[i*2+1, j] if len(h_values)*2 > 1 else axes[1, j]

                df2 = data_dict.get((n,h,'2'))
                df1 = data_dict.get((n,h,'1'))

                if df2 is not None:
                    sns.heatmap(df2, ax=ax_top, annot=False, cmap=custom_cmap, cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=100)
                        
                    for y in separators:
                        ax_top.axhline(y, color='black', linewidth=1.2)
                    # outline around each individual heatmap:
                    for spine in ax_top.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.1)
                        spine.set_edgecolor("gray")
                else:
                    ax_top.axis('off')
                if df1 is not None:
                    sns.heatmap(df1, ax=ax_bot, annot=False, cmap=custom_cmap, cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=100)
                    for y in separators:
                        ax_bot.axhline(y, color='black', linewidth=1.2)
                    # outline around each individual heatmap:
                    for spine in ax_bot.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.1)
                        spine.set_edgecolor("gray")
                else:
                    ax_bot.axis('off')

                # Put "E" and "H" labels on the right side of the heatmaps in the last column
                if j == len(n_values) - 1:
                    # maybe add if clauses in the event that a dataset does not have EDM-1 or EDM-2
                    ax_top.set_ylabel("E", rotation=0, labelpad=20, fontsize=26)
                    ax_top.yaxis.set_label_position("right")
                        
                    ax_bot.set_ylabel("H", rotation=0, labelpad=20, fontsize=26)
                    ax_bot.yaxis.set_label_position("right")
    elif is_mainEff:
        for j, edm in enumerate(sorted(edm_values, reverse=True)):
            for i, h in enumerate(sorted(h_values)):
                ax = axes[j, i]

                # Find the value in the dict corresponding to this n and h (could be either EDM-1 or EDM-2 depending on dataset, but only one of them)
                df = next(
                    data_dict[key] 
                    for key in data_dict 
                    if key[2] == edm and key[1] == h
                )

                if df is not None:
                    sns.heatmap(df, ax=ax, annot=False, cmap=custom_cmap, cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=100)
                    for y in separators:
                        ax.axhline(y, color='black', linewidth=1.2)
                    # outline around each individual heatmap:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.1)
                        spine.set_edgecolor("gray")
                else:
                    ax.axis('off')
                
                if i == 0 and edm == '2': # if first column and the EDM = 2 (E) row
                    ax.set_ylabel("E", rotation=0, labelpad=20, fontsize=30)
                    ax.yaxis.set_label_position("left")
                    ax.yaxis.set_label_coords(-0.1, 0.4)
                elif i == 0 and edm == '1': # if first column and the EDM = 1 (H) row    
                    ax.set_ylabel("H", rotation=0, labelpad=20, fontsize=30)
                    ax.yaxis.set_label_position("left")
                    ax.yaxis.set_label_coords(-0.1, 0.4)

    if is_core2wayEpistasis:
        # Set x-axis labels for Number of Training Instances
        for j, n in enumerate(n_values):
            # Place label centered below the corresponding column (under the last row for that column)
            mid_axs = axes[-1, j] if len(h_values)*2 > 1 else axes[1, j]
            mid_axs.set_xlabel(str(n), fontsize=26)
            mid_axs.xaxis.set_label_coords(0.5, -0.2)  # adjust vertical padding
    elif is_mainEff:
        # Set x-axis labels for Heritability of Model
        for i, h in enumerate(h_values):
            # Place label centered below the corresponding column (under the last row for that column)
            mid_axs = axes[-1, i] if len(edm_values)*2 > 1 else axes[1, i]
            mid_axs.set_xlabel(str(h), fontsize=30)
            mid_axs.xaxis.set_label_coords(0.5, -0.1)  # adjust vertical padding

    if is_core2wayEpistasis:
        # Set y-axis labels for Heritability of Model (once per heritability row)
        for i, h in enumerate(sorted(h_values, reverse=True)):
            # First column only
            ax_top = axes[i*2, 0]
            ax_bot = axes[i*2 + 1, 0]
                
            # Position the label in the middle of top and bottom heatmaps
            mid_y = 0  # normalized vertical coordinate (0 = bottom of ax, 1 = top of ax)
                
            # if there is only one column, can't use set_ylabel twice on the same axis (will overwrite the first one, "E"); so use .text instead
            if len(n_values) == 1:
                ax_top.text(-0.2, mid_y - 0.1, str(h), rotation=0, fontsize=26, va='center', ha='center', transform=ax_top.transAxes)
            else:
                # Use the top subplot to place the label vertically centered
                ax_top.set_ylabel(str(h), rotation=0, fontsize=26)
                ax_top.yaxis.set_label_coords(-0.2, mid_y - 0.1)
    
    # Tight layout with extra spacing
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    # ** LINES BETWEEN INSTANCE/HERITABILITY COMBINATIONS:
    # --- DRAW DIVIDER LINES THROUGH GAP COLUMNS AND ROWS ---
    # Use figure coordinates (0–1 range)
    if is_core2wayEpistasis:
        # Vertical dividers after every column
        for j in range(total_cols - 1):
            # Compute midpoint between the right edge of column j and left edge of next column
            left_bbox = axes[0, j].get_position()
            right_bbox = axes[0, j+1].get_position()
            x_mid = (left_bbox.x1 + right_bbox.x0) / 2
            y_bottom = axes[-1,0].get_position().y0
            y_top = axes[0,0].get_position().y1

            # Vertical line (spanning entire figure)
            line = mlines.Line2D([x_mid, x_mid], [y_bottom, y_top],
                                transform=fig.transFigure, color='black', linewidth=1.5, alpha=0.3)
        
            fig.add_artist(line)

        # Horizontal dividers after every heritability block (every 2 rows)
        for i in range(1, len(h_values)):
            # Get bounding boxes for last heatmap in block i-1 and first in block i
            prev_bottom = axes[(i-1)*2 + 1, 0].get_position().y0  # bottom of bottom subplot of previous block
            next_top = axes[i*2, 0].get_position().y1             # top of top subplot of next block
            y_mid = (prev_bottom + next_top) / 2
            x_left = axes[0,0].get_position().x0
            x_right = axes[0,-1].get_position().x1

            # Horizontal line (spanning entire figure)
            line = mlines.Line2D([x_left, x_right], [y_mid, y_mid],
                                transform=fig.transFigure, color='black', linewidth=1.5, alpha=0.3)
            
            fig.add_artist(line)
        

    outdir = os.path.basename(os.path.normpath(args.basedir))
    parentdir = os.path.dirname(os.path.normpath(args.basedir))
    # save_path = os.path.join(parentdir, outdir, args.prefix + "unified_heatmaps.pdf")
    save_path = os.path.join(parentdir, outdir, args.prefix + "unified_heatmaps_withMuRelief_biggerfont.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Unified heatmap saved to {save_path}")

if __name__ == "__main__":
    main()