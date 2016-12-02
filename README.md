# scikit-rebate

A sklearn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for machine learning.

## Relief-based algorithms

This package contains implementations of the [Relief](https://en.wikipedia.org/wiki/Relief_(feature_selection)) family of feature selection algorithms. **It is still under active development** and we encourage you to check back on this repository regularly for updates.

These algorithms excel at identifying features that are predictive of the outcome in supervised learning problems, and are especially good at identifying feature interactions that are normally overlooked by standard feature selection algorithms.

The main benefit of Relief algorithms is that they identify feature interactions without having to exhaustively check every pairwise interaction, thus taking significantly less time than exhaustive pairwise search.

Relief algorithms are commonly applied to genetic analyses, where epistasis (i.e., feature interactions) is common. However, the algorithms implemented in this package can be applied to almost any supervised classification data set and supports:

* Categorical features

* Continuous features

* Discrete endpoints (i.e., classification)

* Continuous endpoints (i.e., regression)

## License

Please see the [repository license](https://github.com/EpistasisLab/scikit-rebate/blob/master/LICENSE) for the licensing and usage information for scikit-rebate.

Generally, we have licensed scikit-rebate to make it as widely usable as possible.

## Installation

scikit-rebate is built on top of the following existing Python packages:

* NumPy

* SciPy

* scikit-learn

All of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use Python 3 over Python 2 if you're given the choice.

NumPy, SciPy, and scikit-learn can be installed in Anaconda via the command:

```
conda install numpy scipy scikit-learn
```

Once the prerequisites are installed, you should be able to install scikit-rebate with a pip command:

```
pip install skrebate
```

Please [file a new issue](https://github.com/EpistasisLab/scikit-rebate/issues/new) if you run into installation problems.

## Usage

We have designed the Relief algorithms to be integrated directly into scikit-learn machine learning workflows. For example, the ReliefF algorithm can be used as a feature selection step in a scikit-learn pipeline as follows.

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

For more information on the Relief algorithms available in this package, please refer to our documentation. [documentation coming soon]

## Contributing to scikit-rebate

We welcome you to [check the existing issues](https://github.com/EpistasisLab/scikit-rebate/issues/) for bugs or enhancements to work on. If you have an idea for an extension to scikit-rebate, please [file a new issue](https://github.com/EpistasisLab/scikit-rebate/issues/new) so we can discuss it.

## Citing scikit-rebate

If you use scikit-rebate as part of your workflow in a scientific publication, please consider citing the scikit-rebate repository with the following DOI:

[coming soon]
