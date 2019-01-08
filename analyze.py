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

solved_cols = [x for x in benchmark_data_merged.columns.values if 'Solved' in x]

import seaborn as sb
import numpy as np

for problem, problem_group in benchmark_data_merged.groupby('FilePath'):
    problem_group = problem_group.loc[problem_group['AlgorithmShort'].apply(
        lambda x: x not in ['MultiSURF*', 'FixedReliefFPercent20', 'FixedReliefFPercent30',
                            'FixedReliefFPercent38.2',
                            'ReliefF-NN10', 'ReliefF-NN100', 'ReliefF-NN200'])]

    problem_group['AlgorithmShort'] = problem_group['AlgorithmShort'].apply(
        lambda x: x.replace('ANOVAFValue', 'ANOVA F-value')
                   .replace('FixedMultiSURF', 'MultiSURF*') # FixedMultiSURF == MultiSURF*
                   .replace('Fixed', '').replace('chi2', 'Chi^2')
                   .replace('SURFstar', 'SURF*').replace('ReliefFPercent50', 'ReliefF 50% NN')
                   .replace('ReliefFPercent10', 'ReliefF 10% NN').replace('ReliefF-NN100', 'ReliefF 100 NN')
                   .replace('ReliefF-NN10', 'ReliefF 10 NN').replace('RFEExtraTrees', 'RFE ExtraTrees')
                   .replace('MutualInformation', 'Mutual Information'))
    
    algo_labels = ['Random Shuffle',
                   'Chi^2',
                   'ANOVA F-value',
                   'Mutual Information', 
                   'ExtraTrees',
                   'RFE ExtraTrees',
                   'ReliefF 10 NN',
                   'ReliefF 100 NN',
                   'ReliefF 10% NN',
                   'ReliefF 50% NN',
                   'SURF',
                   'SURF*',
                   'MultiSURF*',
                   'MultiSURF'][::-1]
    
    problem_group['AlgorithmShort'] = pd.Categorical(problem_group['AlgorithmShort'],
                                                     algo_labels)

    problem_group_avg = problem_group.groupby(
        'AlgorithmShort')[solved_cols].mean().sort_index(ascending=False)

    plt.figure(figsize=(13, 10))
    
    custom_cmap = sb.color_palette('Oranges', n_colors=1000)[:800] + sb.color_palette('Blues', n_colors=1000)[800:]
    sb.heatmap(data=problem_group_avg.values, vmin=0, vmax=1, cmap=custom_cmap)#'Blues')
    plt.yticks(np.array(range(len(problem_group_avg))) + 0.5, algo_labels[::-1], rotation=0)
    plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90],
               ['Optimal', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], rotation=0)
    plt.xlabel('Predictive features in top % of ranked features')
    plt.title('\n'.join(problem.split('Archive/')[-1].split('/')).replace('_', ' '))
    
    plt.savefig('figures/' + problem.split('Archive/')[-1].replace('/', '_') + '.pdf', bbox_inches='tight')
    plt.savefig('figures/' + problem.split('Archive/')[-1].replace('/', '_') + '.eps', bbox_inches='tight')