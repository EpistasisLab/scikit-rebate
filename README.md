# scikit-rebate

A sklearn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms.

## Relief-based Algorithms

This suite includes stand-alone Python code to run any of
the included/available Relief-based algorithms designed for attribute
filtering and ranking. These algorithms are a quick way to identify
attributes in a dataset that may be most important to predicting some
predefined endpoint. These scripts output an ordered set of attribute
names, along with respective scores (uniquely determined by the particular
algorithm selected). Certain algorithms require key algorithm parameters
to be specified beforehand.
