scikit-rebate is a scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning.

This package contains implementations of the [Relief](https://en.wikipedia.org/wiki/Relief_(feature_selection)) family of feature selection algorithms. **It is still under active development** and we encourage you to check back on this repository regularly for updates.

These algorithms excel at identifying features that are predictive of the outcome in supervised learning problems, and are especially good at identifying feature interactions that are normally overlooked by standard feature selection algorithms.

The main benefit of Relief algorithms is that they identify feature interactions without having to exhaustively check every pairwise interaction, thus taking significantly less time than exhaustive pairwise search.

Relief algorithms are commonly applied to genetic analyses, where epistasis (i.e., feature interactions) is common. However, the algorithms implemented in this package can be applied to almost any supervised classification data set and supports:

* Categorical features

* Continuous features

* Discrete endpoints (i.e., classification)

* Continuous endpoints (i.e., regression)
