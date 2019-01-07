import pandas as pd
import numpy as np
import sys
import csv

csv.field_size_limit(sys.maxsize)
pd.set_option('display.max_columns', None)

benchmark_data = pd.read_csv('benchmark-parsed.tsv.gz', sep='\t', engine='python')
benchmark_data['FilePath'] = benchmark_data['FileName'].apply(lambda x: '/'.join(x.split('/')[:-1]))
benchmark_data.sort_values('FilePath', inplace=True)
print(benchmark_data.head())

summary_sheet = pd.read_excel('SimulatedDataArchiveSummary1.xlsx')
print(summary_sheet.head())

benchmark_data_merged = benchmark_data.merge(summary_sheet, left_on='FilePath', right_on='Specific Path', how='left')
benchmark_data_merged.sort_values(['FilePath', 'Algorithm'], inplace=True)
benchmark_data_merged.loc[benchmark_data_merged['Key Variables'].isnull()]['FilePath'].unique()

print('bd merged')
print(benchmark_data_merged.head())