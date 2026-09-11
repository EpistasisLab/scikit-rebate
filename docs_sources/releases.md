# scikit-rebate 0.8

* Added new core Relief-based algorithms to scikit-rebate: SWRF, SWRF\*, MultiSWRF, MultiSWRF\*, MultiSWRFDB, MultiSWRFDB\*, and μ-Relief.

* Refactored code to speed up runtime by 10-35x, improving the efficiency of the feature scoring process.

* Added the 'categorical_features' hyperparameter, which allows users to manually specify which features should be treated as categorical.

* Added the 'label_type' hyperparameter, which allows users to manually specify whether the target variable (label) is binary, multiclass, or continuous.

* Added the 'multiclass_threshold' hyperparameter, which allows users to specify the maximum number of unique values a target variable can have before it is considered continuous (e.g. multiclass_threshold=10 means a target variable with > 10 unique values is considered continuous).

* Changed the name of the 'discrete_threshold' hyperparameter to 'categorical_threshold'.

* Now utilize np.nanmean instead of np.mean when the amount of missing data is substantial enough to derail computation.

# scikit-rebate 0.7

* The version of scikit-rebate uploaded to pypi (v0.7) and cloned from [https://github.com/EpistasisLab](https://github.com/EpistasisLab), last updated March 20, 2021. Ongoing development of scikit-rebate will take place on the UrbsLab GitHub page.

* Preserves the underlying algorithm and algorithmically the same as scikit-rebate v0.62 on pypi. This release includes the analysis scripts used for the scikit-rebate paper exploring the ability of Relief-based algorithms to detect higher order interactions.

* Fix for the TuRF wrapper algorithm as well as further fixes/updates to the preliminary implementations of VLS and ITER Relief wrapper algorithms.

* As scikit-rebate development continues in the future we will use the UrbsLab/scikit-rebate repo to share code updates, as well as update pypi with the most recent updates continuing on from v0.7.1.

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
