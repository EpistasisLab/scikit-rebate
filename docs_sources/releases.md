# scikit-rebate 0.6

* Fixed internal TuRF implementation so that it outputs scores for all features. Those that make it to the last iteration get true core algorithm scoring, while those that were removed along the way are assigned token scores (lower than the lowest true scoring feature) that indicate when the respective feature(s) were removed. This also alows for greater flexibility in the user specifying the number for features to return. 

* Updated the usage documentation to demonstrate how to use RFE as well as the newly updated internal TuRF implementation. 

* Fixed the pct paramter of TuRF to properly determine the percent of features removed each iteration as well as the total number of iterations as described in the original TuRF paper.  Also managed the edge case to ensure that at least one feature would be removed each TuRF iteration. 

* Fixed ability to parallelize run of core algorithm while using TuRF.

* Updated the unit testing file to remove some excess unite tests, add other relevant ones, speed up testing overall, and make the testing better organized. 

* Added a preliminary implementation of VLSRelief to scikit-rebate, along with associated unit tests. Documentation and code examples not yet supported. 

* Removed some unused code from TuRF implementation.

* Added check in the transform method required by scikit-learn in both relieff.py and turf.py to ensure that the number of selected features requested by the user was not larger than the number of features in the dataset. 

* Reduced the default value for number of features selected

# scikit-rebate 0.5

* Added fixes to score normalizations that should ensure that feature scores for all algorithms fall between -1 and 1. 

* Added multi-class endpoint functionality. (now discriminates between binary and multiclass endpoints) Includes new methods for multi-class score update normalization.

* Fixed normalization for missing data.

* Fixed inconsistent pre-normalization for continuous feature data. 

* Added a custom ramp function to improve performance of all algorithms on data with a mix of discrete and continuous features.  Based on the standard deviation of a given continuous feature. 

* Updated the implementation of TuRF as an internal custom component of ReBATE.

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
