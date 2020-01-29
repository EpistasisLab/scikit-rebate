[scikit-rebate](https://github.com/EpistasisLab/scikit-rebate) is a scikit-learn-compatible Python implementation of ReBATE, a suite of [Relief](https://en.wikipedia.org/wiki/Relief_(feature_selection))-based feature selection algorithms for Machine Learning. As of 5/7/18, **this project is still under active development** and we encourage you to check back on this repository regularly for updates.

These algorithms excel at identifying features that are predictive of the outcome in supervised learning problems, and are especially good at identifying feature interactions that are normally overlooked by standard feature selection methods.

The main benefit of Relief-based algorithms is that they identify feature interactions without having to exhaustively check every pairwise interaction, thus taking significantly less time than exhaustive pairwise search.

Relief-based algorithms are commonly applied to genetic analyses, where epistasis (i.e., feature interactions) is common. However, the algorithms implemented in this package can be applied to almost any supervised classification data set and supports:

* A mix of categorical and/or continuous features

* Data with missing values

* Binary endpoints (i.e., classification)

* Multi-class endpoints (i.e., classification)

* Continuous endpoints (i.e., regression)
