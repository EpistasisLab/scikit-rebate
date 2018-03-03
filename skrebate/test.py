import pandas as pd
from turf import *

genetic_data = pd.read_csv(
    'https://github.com/EpistasisLab/scikit-rebate/raw/master/data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.tsv.gz', sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values
x = TuRF(core_algorithm='MultiSURF')
res = x.fit(features, labels)
