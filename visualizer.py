import pandas as pd
import numpy as np
import sys
import csv
from tqdm import tqdm


csv.field_size_limit(sys.maxsize)
pd.set_option('display.max_columns', None)

benchmark_data = pd.read_csv('benchmark-parsed.tsv.gz', sep='\t', engine='python')
benchmark_data['FilePath'] = benchmark_data['FileName'].apply(lambda x: '/'.join(x.split('/')[:-1]))
benchmark_data.sort_values('FilePath', inplace=True)

summary_sheet = pd.read_excel('SimulatedDataArchiveSummary.xlsx')


benchmark_data_merged = benchmark_data.merge(summary_sheet, left_on='FilePath', right_on='Specific Path', how='left')
benchmark_data_merged.sort_values(['FilePath', 'Algorithm'], inplace=True)
benchmark_data_merged.loc[benchmark_data_merged['Key Variables'].isnull()]['FilePath'].unique()

benchmark_data_merged['NumKeyVariables'] = benchmark_data_merged['Key Variables'].apply(lambda x: str(x).count(',') + 1)

for top_pct in tqdm(list(range(101))[::-1]):
    keep_pct = 1. - (top_pct / 100.)
    benchmark_data_merged['SolvedTop{}Pct'.format(top_pct)] = benchmark_data_merged.apply(
        lambda row: set(row['Key Variables'].split(',')).issubset(
            set(row['FeaturesRanked'].split(',')[
                :max(row['NumKeyVariables'], int(keep_pct * row['Total Features']))])), axis=1)

print(benchmark_data_merged.head())
benchmark_data_merged.to_csv('benchmark-parsed-merged-analyzed.tsv.gz', index=False, compression='gzip', sep='\t')