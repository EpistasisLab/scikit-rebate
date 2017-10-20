Master status: [![Master Build Status](https://travis-ci.org/EpistasisLab/scikit-rebate.svg?branch=master)](https://travis-ci.org/EpistasisLab/scikit-rebate)
[![Master Code Health](https://landscape.io/github/EpistasisLab/scikit-rebate/master/landscape.svg?style=flat)](https://landscape.io/github/EpistasisLab/scikit-rebate/master)
[![Master Coverage Status](https://coveralls.io/repos/github/EpistasisLab/scikit-rebate/badge.svg?branch=master&service=github)](https://coveralls.io/github/EpistasisLab/scikit-rebate?branch=master)

Development status: [![Development Build Status](https://travis-ci.org/EpistasisLab/scikit-rebate.svg?branch=development)](https://travis-ci.org/EpistasisLab/scikit-rebate)
[![Development Code Health](https://landscape.io/github/EpistasisLab/scikit-rebate/development/landscape.svg?style=flat)](https://landscape.io/github/EpistasisLab/scikit-rebate/development)
[![Development Coverage Status](https://coveralls.io/repos/github/EpistasisLab/scikit-rebate/badge.svg?branch=development&service=github)](https://coveralls.io/github/EpistasisLab/scikit-rebate?branch=development)

Package information: ![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.6-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)
[![PyPI version](https://badge.fury.io/py/skrebate.svg)](https://badge.fury.io/py/skrebate)

# scikit-rebate

A scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning.

## Relief-based algorithms

This package contains implementations of the [Relief](https://en.wikipedia.org/wiki/Relief_(feature_selection)) family of feature selection algorithms. **It is still under active development** and we encourage you to check back on this repository regularly for updates.

These algorithms excel at identifying features that are predictive of the outcome in supervised learning problems, and are especially good at identifying feature interactions that are normally overlooked by standard feature selection algorithms.

The main benefit of Relief algorithms is that they identify feature interactions without having to exhaustively check every pairwise interaction, thus taking significantly less time than exhaustive pairwise search.

Relief algorithms are commonly applied to genetic analyses, where epistasis (i.e., feature interactions) is common. However, the algorithms implemented in this package can be applied to almost any supervised classification data set and supports:

* Categorical or continuous features

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

genetic_data = pd.read_csv('https://github.com/EpistasisLab/scikit-rebate/raw/master/data/'
                           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz',
                           sep='\t', compression='gzip')

features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, features, labels)))
>>> 0.795
```

For more information on the Relief algorithms available in this package and how to use them, please refer to our [usage documentation](https://EpistasisLab.github.io/scikit-rebate/using/).

## Contributing to scikit-rebate

We welcome you to [check the existing issues](https://github.com/EpistasisLab/scikit-rebate/issues/) for bugs or enhancements to work on. If you have an idea for an extension to scikit-rebate, please [file a new issue](https://github.com/EpistasisLab/scikit-rebate/issues/new) so we can discuss it.

Please refer to our [contribution guidelines](https://EpistasisLab.github.io/scikit-rebate/contributing/) prior to working on a new feature or bug fix.

## Citing scikit-rebate

If you use scikit-rebate in a scientific publication, please consider citing at least one of the following papers:

Igor Kononenko et al. Overcoming the myopia of inductive learning algorithms with RELIEFF (1997), Applied Intelligence, 7(1), pp. 39-55.

Casey S Greene, Nadia M Penrod, Jeff Kiralis, and Jason H Moore. Spatially Uniform ReliefF (SURF) for computationally-efficient filtering of gene-gene interactions (2009), BioData Mining 2(5).
