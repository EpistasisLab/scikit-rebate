We have designed the Relief algorithms to be integrated directly into scikit-learn machine learning workflows. Below, we provide code samples showing how the various Relief algorithms can be used in a scikit-learn pipeline.

## ReliefF

ReliefF is the most basic of the Relief-based algorithms, and it requires you to specify

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz', sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))
>>> 0.795
```

## SURF



## SURF*



## MultiSURF



## TURF

