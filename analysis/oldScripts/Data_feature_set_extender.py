# %%
import os
import pandas as pd
import numpy as np

# %%
def hardy_weinberg_equilibrium(freq):
    p = np.sqrt(freq)
    q = 1 - p
    return np.random.choice([0, 1, 2], p=[q**2, 2*p*q, p**2])

# %%
def get_max_random_feature(column_names):
    column_names = [x for x in column_names if x != 'Class' and x.startswith('N')]
    column_names = [x[1:] if x.startswith('N') else x for x in column_names]
    max_value = max(int(x) for x in column_names)
    return max_value
    

# %%
#total_features = 40  #20, 40, 60, 80, 100
feature_list = [40,60,80]
completed = [20,100]
min_MAF = 0.01
max_MAF = 0.5
folder_path = "../Benchmark Data"


# %%
#go through features to generate list one at a time

def process_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        print(root)
        if 'a_20' in root:
            print(root)
            for total_features in feature_list:
                print(total_features)
            
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)

                        # Load the dataset
                        df = pd.read_csv(file_path, sep='\t')
                        column_names = df.columns.tolist()
                        max_value = get_max_random_feature(column_names)
                        num_columns = df.shape[1]
                        n = total_features +1 - num_columns

                        # Generate a random minor allele frequency between 0.05 and 0.5
                        freq = np.random.uniform(min_MAF, max_MAF)

                        # Add n new columns with values 0, 1, or 2 based on Hardy-Weinberg equilibrium
                        for i in range(n):
                            col_name = 'N'+str(i+max_value+1)
                            df[col_name] = df.apply(lambda _: hardy_weinberg_equilibrium(freq), axis=1)

                        new_root = root.replace('a_20', 'a_'+str(total_features))

                        if not os.path.exists(new_root):
                            os.makedirs(new_root)

                        new_file_path = file_path.replace('a_20', 'a_'+str(total_features))

                        # Save the modified dataset
                        df.to_csv(new_file_path, sep='\t', index=False)

# %%
process_files_in_folder(folder_path)


