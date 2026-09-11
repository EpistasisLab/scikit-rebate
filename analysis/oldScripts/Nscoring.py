# %%
import os
import pandas as pd
import re

# Function to process results directory and return a pivot table with feature importance
def process_results_directory(root_dir, meaningful_parts):
    all_data = []  # List to store all data
    
    # Walk through the directory tree
    for subdir, dirs, files in os.walk(root_dir):
        # Check if subdir contains 'ABS' or 'MutualInfo'
        if 'ABS' in subdir or 'MutualInfo' in subdir:
            if subdir != root_dir:  # Ensure not processing the root directory
                method = subdir.split(os.sep)[-1].replace('ABS_', '')  # Extract method name from subdir
                file_averages = []  # List to store average importances for each file
                for file in files:
                    if file.endswith('.txt'):  # Process only .txt files
                        file_path = os.path.join(subdir, file)  # Construct file path
                        try:
                            data = pd.read_csv(file_path, delimiter='\t')  # Read data from file
                            n_data = data[data['Feature'].str.startswith('N')]  # Filter data for features starting with 'N'
                            importance_cols = [col for col in data.columns if "Feature_Importance" in col]  # Find importance columns

                            # Check if there are 'N' features and importance columns
                            if not n_data.empty and importance_cols:
                                # Calculate average importance for 'N' features within the current file
                                for col in importance_cols:
                                    avg_importance = n_data[col].mean()  # Calculate average importance
                                    importance_type = col.replace('Feature_Importance', '').strip()  # Extract importance type
                                    method_column_name = f"{method}_{importance_type}".rstrip('_')  # Create method column name
                                    
                                    entry = {
                                        'Feature': 'average_N',  # Use 'average_N' as the feature name
                                        'Method': method_column_name,  # Method and importance type
                                        'Importance': avg_importance,  # Average importance value within the file
                                        **meaningful_parts  # Add meaningful parts
                                    }
                                    file_averages.append(entry)  # Add entry to file_averages
                            else:
                                print(f"No 'N' features or importance columns in {file_path}")

                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")  # Print error if reading fails

                if file_averages:
                    # Aggregate the file-level averages for the current method
                    df_file_averages = pd.DataFrame(file_averages)
                    method_average = df_file_averages.groupby(['Feature', 'Method', *meaningful_parts.keys()]).mean().reset_index()
                    all_data.extend(method_average.to_dict('records'))  # Add aggregated data to all_data
                else:
                    print(f"No valid data found in {subdir}")

    if all_data:
        results_df = pd.DataFrame(all_data)  # Create DataFrame from all_data
        pivot_df = results_df.pivot_table(index=['Feature', *meaningful_parts.keys()], columns='Method', values='Importance', aggfunc='mean').reset_index()
        # Get the current non-Method columns
        non_method_cols = ['Feature', *meaningful_parts.keys()]
        # Get the current Method columns and sort them as required
        method_cols = [col for col in pivot_df.columns if col not in non_method_cols]
        # sorted_method_cols = ['ReliefF10', 'ReliefF10_ABS', 'ReliefF', 'ReliefF_ABS', 'MultiSURF', 'MultiSURF_ABS', 'MultiSURFstar', 'MultiSURFstar_ABS', 'MutualInformation']
        sorted_method_cols = ['SURFstar', 'SURFstar_ABS', 'MultiSURFstar', 'MultiSURFstar_ABS', 'SWRFstar','SWRFstar_ABS','MutualInfo']
        
        # Combine the non-Method columns with the sorted Method columns
        pivot_df = pivot_df[non_method_cols + sorted_method_cols]
        return pivot_df  # Return the pivot table
    else:
        print("No data collected from any files")
        return pd.DataFrame()  # Return an empty DataFrame if no data

def extract_numeric_parts(s):
    """Extract numeric parts from a string and return as a tuple of integers."""
    return tuple(map(int, re.findall(r'\d+', s)))

def find_and_process_results(start_dir):
    master_df = pd.DataFrame()  # Initialize an empty DataFrame to store the master results
    for root, dirs, files in os.walk(start_dir):  # Walk through the directory tree
        if 'Results' in dirs:  # Check if 'Results' directory exists in the current directory
            path_parts = os.path.relpath(root, start_dir).split(os.sep)  # Get the relative path parts
            meaningful_parts = {}  # Dictionary to store meaningful parts
            for part in path_parts:  # Iterate over path parts - will need to update this for new descriptors if needed
                if 'xor_' in part:
                    meaningful_parts['X1'] = part  # Order of epistasis with additional descriptors
                elif 'a_' in part:
                    meaningful_parts['X2'] = part  # Feature count

            results_dir = os.path.join(root, 'Results')  # Construct the path to the 'Results' directory
            print(f"Processing 'Results' folder at: {results_dir}")
            results_df = process_results_directory(results_dir, meaningful_parts)  # Process the results directory
            if not results_df.empty:  # If the results DataFrame is not empty
                csv_path = os.path.join(results_dir, 'N_average_feature_importance.csv')  # Path to save the results CSV
                results_df.to_csv(csv_path, index=False)  # Save the results DataFrame to a CSV file
                print(f"Average feature importance results saved to {csv_path}")

                # Create a numeric tuple for sorting X1
                try:
                    results_df['Value'] = results_df['X1'].apply(extract_numeric_parts)  # Extract numeric parts from X1
                except Exception as e:
                    results_df['Value'] = results_df['X2'].apply(extract_numeric_parts)
                    # results_df['X1_numeric'] = results_df['X2_numeric']  # Fallback to X2 if X1 extraction fails
                
                # Append to the master DataFrame
                master_df = pd.concat([master_df, results_df], ignore_index=True)
            else:
                print(f"No data found in 'Results' folder at: {results_dir}")

    # Sort the master DataFrame
    if not master_df.empty:  # If the master DataFrame is not empty
        master_df = master_df.sort_values(by=['Value', 'Feature'])  # Sort by numeric parts of X1 and Feature
        master_df.drop('Value', axis=1, inplace=True)  # Remove the auxiliary column after sorting

        # Calculate grand average for all numeric columns and append it as the last row
        numeric_cols = master_df.select_dtypes(include='number').columns
        grand_average = master_df[numeric_cols].mean().to_frame().T
        for col in master_df.columns:
            if col not in numeric_cols:
                grand_average[col] = 'Grand_Average'
        master_df = pd.concat([master_df, grand_average], ignore_index=True)

        master_csv_path = os.path.join(start_dir, 'N_master_feature_importance.csv')  # Path to save the master CSV
        master_df.to_csv(master_csv_path, index=False)  # Save the master DataFrame to a CSV file
        print(f"Master feature importance results saved to {master_csv_path}")  # Print confirmation
    else:
        print("No master data to save")

# Start the process from the current working directory
find_and_process_results('../core2wayEpistasis')


