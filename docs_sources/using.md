We have designed the Relief algorithms to be integrated directly into scikit-learn machine learning workflows. Below, we provide code samples showing how the various Relief algorithms can be used as feature selection methods in scikit-learn pipelines.

For details on the algorithmic differences between the various Relief algorithms, please refer to [this research paper](https://biodatamining.biomedcentral.com/articles/10.1186/1756-0381-2-5).

## ReliefF

ReliefF is the most basic of the Relief-based feature selection algorithms, and it requires you to specify the number of nearest neighbors to consider in the scoring algorithm. The parameters for the ReliefF algorithm are as follows:

<table>
<tr>
<th>Parameter</th>
<th width="15%">Valid values</th>
<th>Effect</th>
</tr>
<tr>
<td>n_features_to_select</td>
<td>Any positive integer</td>
<td>The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the ReliefF scoring process.</td>
</tr>
<tr>
<td>n_neighbors</td>
<td>Any positive integer</td>
<td>The number of nearest neighbors to consider in the ReliefF feature scoring process. Generally the more neighbors the algorithm considers, the better the scores are. However, considering more neighbors makes the algorithm run slower.</td>
</tr>
<tr>
<td>discrete_limit</td>
<td>Any positive integer</td>
<td>Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > discrete_threshold, then it is considered continuous, or discrete otherwise.</td>
</tr>
<tr>
<td>n_jobs</td>
<td>Any positive integer or -1</td>
<td>The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores. Currently not supported in Python 2.</td>
</tr>
</table>

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))
>>> 0.795
```

## SURF

SURF, SURF*, and MultiSURF are all extensions to the ReliefF algorithm that automatically determine the ideal number of neighbors to consider when scoring the features.

<table>
<tr>
<th>Parameter</th>
<th width="15%">Valid values</th>
<th>Effect</th>
</tr>
<tr>
<td>n_features_to_select</td>
<td>Any positive integer</td>
<td>The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the SURF scoring process.</td>
</tr>
<tr>
<td>discrete_limit</td>
<td>Any positive integer</td>
<td>Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > discrete_threshold, then it is considered continuous, or discrete otherwise.</td>
</tr>
<tr>
<td>n_jobs</td>
<td>Any positive integer or -1</td>
<td>The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores. Currently not supported in Python 2.</td>
</tr>
</table>

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import SURF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

clf = make_pipeline(SURF(n_features_to_select=2),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))
>>> 0.795
```

## SURF*

<table>
<tr>
<th>Parameter</th>
<th width="15%">Valid values</th>
<th>Effect</th>
</tr>
<tr>
<td>n_features_to_select</td>
<td>Any positive integer</td>
<td>The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the SURF* scoring process.</td>
</tr>
<tr>
<td>discrete_limit</td>
<td>Any positive integer</td>
<td>Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > discrete_threshold, then it is considered continuous, or discrete otherwise.</td>
</tr>
<tr>
<td>n_jobs</td>
<td>Any positive integer or -1</td>
<td>The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores. Currently not supported in Python 2.</td>
</tr>
</table>

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import SURFstar
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

clf = make_pipeline(SURFstar(n_features_to_select=2),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))
>>> 0.795
```

## MultiSURF

<table>
<tr>
<th>Parameter</th>
<th width="15%">Valid values</th>
<th>Effect</th>
</tr>
<tr>
<td>n_features_to_select</td>
<td>Any positive integer</td>
<td>The number of best features to retain after the feature selection process. The "best" features are the highest-scored features according to the MultiSURF scoring process.</td>
</tr>
<tr>
<td>discrete_limit</td>
<td>Any positive integer</td>
<td>Value used to determine if a feature is discrete or continuous. If the number of unique levels in a feature is > discrete_threshold, then it is considered continuous, or discrete otherwise.</td>
</tr>
<tr>
<td>n_jobs</td>
<td>Any positive integer or -1</td>
<td>The number cores to dedicate to running the algorithm in parallel with joblib. Set to -1 to use all available cores. Currently not supported in Python 2.</td>
</tr>
</table>

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import MultiSURF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

clf = make_pipeline(MultiSURF(n_features_to_select=2),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))
>>> 0.795
```

## TURF

TURF advances the feature selection process from a single round to a multi-round process, and can be used in conjunction with any of the Relief-based algorithms. TURF begins with all of the features in the first round, scores them using one of the Relief-based algorithms, then eliminates a portion of them that have the worst scores. With this reduced feature set, TURF again scores the remaining features and eliminates a portion of the worst-scoring features. This process is repeated until a predefined number of features remain.

Essentially, TURF is equivalent to [Recursive Feature Elimination](http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination), as [implemented](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) in scikit-learn. Below is a code sample of how RFE can be used in conjunction with the Relief-based algorithms.

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

clf = make_pipeline(RFE(ReliefF(), n_features_to_select=2),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))
>>> 0.795
```

## Acquiring feature importance scores

In some cases, it may be useful to compute feature importance scores without actually performing feature selection. We have made it possible to access all Relief-based algorithm's scores via the `feature_importances_` attribute. Below is a code example showing how to access the scores from the ReliefF algorithm.

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.model_selection import train_test_split

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Make sure to compute the feature importance scores from only your training set
X_train, X_test, y_train, y_test = train_test_split(features, labels)

fs = ReliefF()
fs.fit(X_train, y_train)

for feature_name, feature_score in zip(genetic_data.drop('class', axis=1).columns,
                                       fs.feature_importances_):
    print(feature_name, '\t', feature_score)

>>>N0 	 -0.0000166666666667
>>>N1 	 -0.006175
>>>N2 	 -0.0079
>>>N3 	 -0.006275
>>>N4 	 -0.00684166666667
>>>N5 	 -0.0104416666667
>>>N6 	 -0.010275
>>>N7 	 -0.00785
>>>N8 	 -0.00824166666667
>>>N9 	 -0.00515
>>>N10 	 -0.000216666666667
>>>N11 	 -0.0039
>>>N12 	 -0.00291666666667
>>>N13 	 -0.00345833333333
>>>N14 	 -0.00324166666667
>>>N15 	 -0.00886666666667
>>>N16 	 -0.00611666666667
>>>N17 	 -0.007325
>>>P1 	 0.108966666667
>>>P2 	 0.111
```

Higher positive scores indicate that the features are likely predictive of the outcome, whereas negative scores indicate that the features are likely noise. We generally recommend removing all features with negative scores at a minimum.
