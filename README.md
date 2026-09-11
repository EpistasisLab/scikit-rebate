<!--Master status: [![Master Build Status](https://travis-ci.org/UrbsLab/scikit-rebate.svg?branch=master)](https://travis-ci.org/UrbsLab/scikit-rebate)
[![Master Code Health](https://landscape.io/github/UrbsLab/scikit-rebate/master/landscape.svg?style=flat)](https://landscape.io/github/EpistasisLab/scikit-rebate/master)
[![Master Coverage Status](https://coveralls.io/repos/github/UrbsLab/scikit-rebate/badge.svg?branch=master&service=github)](https://coveralls.io/github/UrbsLab/scikit-rebate?branch=master)

Development status: [![Development Build Status](https://travis-ci.org/UrbsLab/scikit-rebate.svg?branch=development)](https://travis-ci.org/UrbsLab/scikit-rebate)
[![Development Code Health](https://landscape.io/github/UrbsLabLab/scikit-rebate/development/landscape.svg?style=flat)](https://landscape.io/github/UrbsLab/scikit-rebate/development)
[![Development Coverage Status](https://coveralls.io/repos/github/UrbsLab/scikit-rebate/badge.svg?branch=development&service=github)](https://coveralls.io/github/UrbsLab/scikit-rebate?branch=development)
-->
Package information: ![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg) ![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)
[![PyPI version](https://img.shields.io/pypi/v/skrebate.svg)](https://pypi.org/project/skrebate/)
<!--[![PyPI version](https://badge.fury.io/py/skrebate.svg)](https://badge.fury.io/py/skrebate)-->

# scikit-rebate

This package includes a scikit-learn-compatible Python implementation of ReBATE, a suite of [Relief-based feature selection algorithms](<https://en.wikipedia.org/wiki/Relief_(feature_selection)>) (RBAs) for machine learning. These Relief-based algorithms (RBAs) are designed for feature weighting/selection as part of a machine learning pipeline (supervised learning). These algorithms offer a computationally efficient way to perform feature selection that is sensitive to feature interactions as well as simple univariate associations, unlike most currently available filter-based feature selection methods. The main benefit of Relief-based algorithms is that they identify feature interactions without having to exhaustively check every pairwise interaction, thus taking significantly less time than exhaustive pairwise search.

Relief-based algorithms can be applied to almost any structured, labeled tabular dataset for supervised learning. For example, these algorithms have commonly been applied to genetic/genomic analyses given that can detect features involved in epistasis (i.e. feature interactions) and they scale linearly with number of features (however note that they scale quadratically number of training instances). 

We refer to the different RBA options as 'core' algorithms which presently include Relief, ReliefF, SURF, SURF\*, MultiSURF\*, MultiSURF, SWRF\*, SWRF, MultiSWRF\*, MultiSWRF, MultiSWRFDB\*, MultiSWRFDB, and μ-Relief. Core algorithms are generally very reliable at detecting pure 2-way interactions (heritability of 0.4) in datasets with up to 5000-10000 features. Additionally, implementations of RBA 'wrapper' algorithms are currently available, including TuRF, VLS, and a combination of TuRF+VLS. These wrapper algorithms help RBAs scale up their sensitivity to even pure interactions in datasets with > 10000 features to at least 100K features.  

Importantly, all RBA algorithms in this package support datasets with the following characteristics:

- Feature sets that are discrete/categorical, continuous-valued or a mix of both 

- Data with missing values

- Binary endpoints (i.e., classification)

- Multi-class endpoints (i.e., classification)

- Continuous endpoints (i.e., regression)

Of note, certain core algorithms have hyperparameter options that users can specify (beyond default settings), e.g. ReliefF’s parameter for ‘k’ number of nearest neighbors. These packages 'automatically detect these relevant characteristics from loaded data. However, when it comes to treating features appropriately as either categorical vs. quantiative features we recommend users specify feature types (using the `categorical_features` hyperparameter) rather than relying on `categorical_threshold` hyperparameter which can be inprecise in assigning feature types. 

Full documentation for this package is available at: [usage documentation](https://urbslab.github.io/scikit-rebate/using/).

**This reposistory is still under active development**, thus we encourage you to notify us if you discovery any bugs as well as check this respository for package updates.
<!-- Certain algorithms require user specified run parameters (e.g. ReliefF requires the user to specify some 'k' number of nearest neighbors).  -->

<!--An alternative 'stand-alone' version of [ReBATE](https://github.com/EpistasisLab/ReBATE) is also available that focuses on improving run-time with the use of Cython for optimization. This implementation also outputs feature names and associated feature scores as a text file by default. -->

## License

Please see the [repository license](https://github.com/UrbsLab/scikit-rebate/blob/master/LICENSE) for the licensing and usage information for scikit-rebate.

We have licensed scikit-rebate to make it as widely usable as possible.

## Installation

scikit-rebate is built on top of the following existing Python packages:

- NumPy

- SciPy

- scikit-learn

All of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use Python 3 over Python 2 if you're given the choice.

NumPy, SciPy, and scikit-learn can be installed in Anaconda via the command:

```
conda install numpy scipy scikit-learn
```

Once the prerequisites are installed, you should be able to install scikit-rebate with a pip command:

```
pip install skrebate
```

Please [file a new issue](https://github.com/UrbsLab/scikit-rebate/issues/new) if you run into installation problems.

## Usage

### Basic Usage

To use an algorithm from ReBATE as a feature selection method:

```Python
# Import necessary packages
import pandas as pd
from skrebate import ReliefF

# Load the example dataset
genetic_data = pd.read_csv(
    './data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv')

# Separate the features and labels from the dataset
features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Apply the ReliefF algorithm for feature selection
fs = ReliefF()
fs.fit(features, labels)

# Print out the results
feature_name = genetic_data.drop('class', axis=1).columns
fs.summary(feature_name=feature_name)

>>> Feature name   Feature importances    Feature rank
>>> P2             0.12330000             1
>>> P1             0.11892500             2
>>> N0             -0.00018125            3
>>> N10            -0.00075625            4
>>> N13            -0.00320625            5
>>> N14            -0.00402500            6
>>> N4             -0.00582500            7
>>> N1             -0.00595000            8
>>> N8             -0.00653750            9
>>> N12            -0.00696250            10
>>> N16            -0.00705000            11
>>> N17            -0.00740625            12
>>> N5             -0.00788750            13
>>> N11            -0.00822500            14
>>> N9             -0.00826250            15
>>> N2             -0.00871875            16
>>> N3             -0.00872500            17
>>> N7             -0.00991875            18
>>> N6             -0.01038750            19
>>> N15            -0.01044375            20
```

### Using as End-to-end Pipeline

We have designed the Relief-based algorithms to be integrated directly into scikit-learn machine learning workflows. For example, the ReliefF algorithm can be used as a feature selection step in a scikit-learn pipeline as follows:

```Python
# Import necessary packages
import pandas as pd
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the example dataset
genetic_data = pd.read_csv(
    './data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv')

# Separate the features and labels from the dataset
features, labels = genetic_data.drop('class', axis=1).values, genetic_data['class'].values

# Split the data to training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Make pipeline
clf = make_pipeline(
    ReliefF(n_features_to_select=2),
    RandomForestClassifier(n_estimators=100)
)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model on testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
>>> Accuracy: 0.781
```

<!-- For more information on the Relief-based algorithms available in this package and how to use them, please refer to our [usage documentation](https://EpistasisLab.github.io/scikit-rebate/using/).

For updated documentation in this forked repository, please refer to this [updated usage documentation](https://urbslab.github.io/scikit-rebate/using/) -->



## Contributing to scikit-rebate

We welcome you to [check the existing issues](https://github.com/UrbsLab/scikit-rebate/issues/) for bugs or enhancements to work on. If you have an idea for an extension to scikit-rebate, please [file a new issue](https://github.com/UrbsLab/scikit-rebate/issues/new) so we can discuss it.

Please refer to our [contribution guidelines](https://UrbsLab.github.io/scikit-rebate/contributing/) prior to working on a new feature or bug fix.

## Citing scikit-rebate
Please note that a new manuscript on this updated ReBATE package has been recently submitted for publication.  In the meantime, if you use scikit-rebate in a scientific publication, please consider citing the following paper:

Ryan J. Urbanowicz, Randal S. Olson, Peter Schmitt, Melissa Meeker, Jason H. Moore (2018). Benchmarking Relief-Based Feature Selection Methods for Bioinformatics Data Mining. _Journal of Biomedical Informatics_, 85, 168-188. DOI: [10.1016/j.jbi.2018.07.015](https://doi.org/10.1016/j.jbi.2018.07.015)

### BibTeX entry:

```bibtex
@article{Urbanowicz2018Benchmarking,
  author    = {Urbanowicz, Ryan J. and Olson, Randal S. and Schmitt, Peter and Meeker, Melissa and Moore, Jason H.},
  title     = {Benchmarking Relief-Based Feature Selection Methods for Bioinformatics Data Mining},
  journal   = {Journal of Biomedical Informatics},
  volume    = {85},
  pages     = {168--188},
  year      = {2018},
  doi       = {10.1016/j.jbi.2018.07.015},
  url       = {https://doi.org/10.1016/j.jbi.2018.07.015}
}
```
