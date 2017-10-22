# scikit-rebate 0.4

* Added support for multicore processing to all Relief algorithms. Multiprocessing is now also supported in Python 2.

* The `ReliefF` algorithm now accepts float values in the range (0, 1.0] for the `n_neighbors` parameter. Float values will be interpreted as a fraction of the training set sample size.

* Refined the MultiSURF and MultiSURF* algorithms. From our internal research, MultiSURF is now one of our best-performing feature selection algorithms.

# scikit-rebate 0.3

* Added a parallelization parameter, `n_jobs`, to ReliefF, SURF, SURF*, and MultiSURF via joblib.

* Renamed the `dlimit` parameter to `discrete_limit` to better reflect the purpose of the parameter.

* Minor code optimizations.

# scikit-rebate 0.2

* Added documentation.

* Minor code optimizations.

# scikit-rebate 0.1

* Initial release of Relief algorithms, including ReliefF, SURF, SURF*, and MultiSURF.
