# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ### Below is the parsing code that goes through all result files and calculates the ranks for each feature for all replicates

# %%
# Define the path to the 'Results' directory
results_dir = '../core2wayEpistasis/s_200/her_0.1__maf_0.2/a_100s_200her_0.1__maf_0.2_EDM-1/Results' # Change this to the path of your 'Results' directory

# Initialize a DataFrame to hold all rankings
all_rankings_df = pd.DataFrame()

# Initialize a dictionary to hold the total number of features (N) for each RBA method
total_features_per_rba = {}

# Iterate over each subfolder in the Results directory, each named for an RBA
for rba in os.listdir(results_dir):
    rba_path = os.path.join(results_dir, rba)
    if os.path.isdir(rba_path):  # Ensure it's a directory
        method_feature_counts = []  # To store feature counts for each dataset within this method
        for file in os.listdir(rba_path):
            if file.endswith('.txt'):  # Ensure the file is a .txt file
                file_path = os.path.join(rba_path, file)
                
                # Extract the dataset identifier from the file name
                parts = file.split('_')
                dataset_id = '_'.join(parts[-2].split('_')[:2])
                
                # Determine Ranks
                df = pd.read_csv(file_path, sep='\t', usecols=['Feature'])
                if rba == "RandomShuffle":
                    df['Rank'] = df.index + 1
                else:
                    column_to_use = 'ABS_Feature_Importance' if "ABS" in rba else 'Feature_Importance'
                    df = pd.read_csv(file_path, sep='\t', usecols=['Feature', column_to_use])
                    df.sort_values(by=column_to_use, ascending=False, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    df['Rank'] = df.index + 1
                
                # Store in a dataframe
                predictive_df = df[df['Feature'].str.startswith('M')][['Feature', 'Rank']]
                predictive_df['RBA'] = rba
                predictive_df['Dataset'] = dataset_id
                
                all_rankings_df = pd.concat([all_rankings_df, predictive_df], ignore_index=True)
                
                # Add the feature count of this dataset to the list for this method
                method_feature_counts.append(df['Feature'].nunique())
        
        # Store the maximum feature count encountered for this method as N
        total_features_per_rba[rba] = max(method_feature_counts)

# Now, you have both the rankings and the total number of features (N) for each RBA method
# You can use total_features_per_rba to access N for each method as needed

# Specify the path to save the consolidated rankings file
save_path = os.path.join(results_dir, 'consolidated_rankings.csv')
all_rankings_df.to_csv(save_path, index=False)

print(f"Consolidated rankings saved to: {save_path}")


# %% [markdown]
# ### This code generates the ranking percentages that are needed to create the heatmaps

# %%
# Assuming all_rankings_df is correctly prepared and contains 'RBA', 'Feature', 'Rank', 'Dataset'

# Since all datasets have the same N, we can pick the N from any RBA method from total_features_per_rba
N = next(iter(total_features_per_rba.values()))

# Step 1: Identify the lowest-ranked predictive feature for each dataset for each RBA
lowest_ranks = all_rankings_df.groupby(['RBA', 'Dataset'])['Rank'].max().reset_index()

# Initialize a structure to hold the calculated percentages for each RBA method
percentages = {rba: [0] * N for rba in lowest_ranks['RBA'].unique()}

# Step 2: Calculate percentages for each position for each RBA
for rba in percentages.keys():
    rba_data = lowest_ranks[lowest_ranks['RBA'] == rba]
    for position in range(1, N + 1):
        # Count how many of the lowest ranks are better (lower number) than the current position
        count_higher = rba_data[rba_data['Rank'] <= (position)].shape[0]
        total_datasets = rba_data.shape[0]  # Should be 30 per RBA if there are 30 datasets
        percentages[rba][position - 1] = (count_higher / total_datasets) * 100

# Convert the percentages to a DataFrame for visualization
percentages_df = pd.DataFrame(percentages, index=range(1, N + 1))

# Save percentages_df to a CSV file
save_path_percentages = os.path.join(results_dir, 'percentages_df.csv')
percentages_df.to_csv(save_path_percentages, index_label='Ranking Position')

print("Percentages saved to percentages_df.csv.")


# %% [markdown]
# ### Plotting code for heatmaps

# %%
# Create custom color scheme
custom_cmap = sns.color_palette('Oranges', n_colors=1000)[:800] + sns.color_palette('Blues', n_colors=1000)[800:]

# Define your preferred order of the RBAs as a list
rba_order = [
    'RandomShuffle',
    'MutualInfo',
    'SURFstar',
    'MultiSURFstar',
    'SWRFstar',
    'ABS_SURFstar',
    'ABS_MultiSURFstar',
    'ABS_SWRFstar',
]

# Define a mapping from your RBA order to new descriptive names
rba_descriptive_names = {
    'RandomShuffle': 'Random Shuffle',
    'MutualInfo': 'Mutual Information',
    'SURFstar': 'SURF*',
    'SWRFstar': 'SWRF*',
    'MultiSURFstar': 'MultiSURF*',
    'ABS_SURFstar': 'SURF* ABS',
    'ABS_MultiSURFstar': 'MultiSURF* ABS',
    'ABS_SWRFstar': 'SWRF* ABS',
}

percentages_df = percentages_df.iloc[1:] # Drop the first row as this will always be 0
percentages_df_transposed = percentages_df.T # Transpose percentages_df to switch rows and columns for horizontal orientation

# Reorder the DataFrame according to your defined RBA order
percentages_df_ordered = percentages_df_transposed.loc[rba_order] # The .loc indexer reindexes the DataFrame to the specified order; any missing labels will result in NaN rows

# Define the tick labels as percentages of optimality
xtick_labels = ['Optimal', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
# Generate a list of positions at which to place the x-tick labels, assuming they should be placed at even intervals
xtick_positions = np.linspace(start=0, stop=percentages_df_transposed.shape[1] - 0.13, num=len(xtick_labels))

# Create the heatmap with the reordered DataFrame
plt.figure(figsize=(12, 7))  # Adjust the size as necessary
heatmap = sns.heatmap(percentages_df_ordered, annot=False, fmt=".1f", cmap=custom_cmap, cbar_kws={'label': 'Power (Frequency of Success)'})

# Add horizontal lines manually between Methods
for i in range(percentages_df_ordered.shape[0] - 1):
    heatmap.axhline(i + 1, color='black', linewidth=1.5)

# Adding black border around the heatmap
for _, spine in heatmap.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_edgecolor("black")

# Adding black border around the color bar in legend
cbar = heatmap.collections[0].colorbar
cbar.outline.set_linewidth(1.5)
cbar.outline.set_edgecolor("black")

dataset_id = results_dir.split('/')[-2]  # Extract the dataset identifier from the results directory path

# Set the title and axis labels appropriately
heatmap.set_title('core2wayEpi_ABS_' + dataset_id, fontsize = 16)
heatmap.set_xlabel('Predictive features in top % of ranked features', fontsize = 14)
heatmap.set_ylabel('Method', fontsize = 14)

# Set the custom x-tick labels, positions, and fontsize
heatmap.set_xticks(xtick_positions)
heatmap.set_xticklabels(xtick_labels, rotation=0, fontsize=11)

# Set the custom y-tick labels fontsize
new_ytick_labels = [rba_descriptive_names[rba] for rba in rba_order]
heatmap.set_yticklabels(new_ytick_labels, fontsize=11)

# Save and show the Plot
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42
plt.tight_layout()  # Adjust the layout
save_path = os.path.join(results_dir, 'core2wayEpi_ABS_' + dataset_id + '.pdf') # Save
plt.savefig(save_path, format='pdf', bbox_inches='tight') # Save
plt.show()  # Display the heatmap

# %%



