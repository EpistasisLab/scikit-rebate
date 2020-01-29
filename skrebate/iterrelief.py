import numpy as np
import pandas as pd
import time
import warnings
import sys
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
# from sklearn.feature_selection.base import SelectorMixin
from sklearn.externals.joblib import Parallel, delayed
# from .scoring_utils import get_row_missing, ReliefF_compute_scores
# from multisurf import MultiSURF
# from multisurfstar import MultiSURFstar
# from surf import SURF
# from surfstar import SURFstar
from .multisurf import MultiSURF
from .multisurfstar import MultiSURFstar
from .surf import SURF
from .surfstar import SURFstar
from .relieff import ReliefF


class IterRelief(BaseEstimator, TransformerMixin):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, core_algorithm, weight_flag=1, n_features_to_select=2, n_neighbors=100, max_iter=10, discrete_threshold=10, verbose=False, n_jobs=1):
        """Sets up IterRelief to perform feature selection.

        Parameters
        ----------
        core_algorithm: Core Relief Algorithm to perform VLSRelief iterations on
        weight_flag: int (default: 1)
            flag to determine whether weight update performed on distance measure alone (weight_flag=1)
            or on both distance and score (weight_flag = 2)
        n_features_to_select: int (default: 10)
            the number of top features (according to the relieff score) to
            retain after feature selection is applied.
        max_iter: int (default 10)
            the maximum number of iterations to be performed.
        discrete_threshold: int (default: 10)
            Value used to determine if a feature is discrete or continuous.
            If the number of unique levels in a feature is > discrete_threshold, then it is
            considered continuous, or discrete otherwise.
        verbose: bool (default: False)
            If True, output timing of distance array and scoring
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.

        """
        self.core_algorithm = core_algorithm
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.weight_flag = weight_flag
        self.max_iter = max_iter

    #=========================================================================#
    def fit(self, X, y):
        """
        Runs a core relief algorithm to generate feauture importance scores. Then these
        scores are used to update the distance vector and the scores for the next iteration.
        The process repeats till a user defined number of iterations has been run.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        Returns
        -------
        Copy of the IterRelief instance
        """

        self.X_mat = X
        self._y = y
        #self.headers = headers

        # Combine IterRelief with specified 'core' Relief-based algorithm
        if self.core_algorithm.lower() == "multisurf":
            core = MultiSURF(n_features_to_select=self.n_features_to_select,
                             discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        elif self.core_algorithm.lower() == "multisurfstar":
            core = MultiSURFstar(n_features_to_select=self.n_features_to_select,
                                 discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        elif self.core_algorithm.lower() == "surf":
            core = SURF(n_features_to_select=self.n_features_to_select,
                        discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        elif self.core_algorithm.lower() == "surfstar":
            core = SURFstar(n_features_to_select=self.n_features_to_select,
                            discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        elif self.core_algorithm.lower() == "relieff":
            core = ReliefF(n_features_to_select=self.n_features_to_select, n_neighbors=self.n_neighbors,
                           discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        # Determine total number of features
        total_num_features = X.shape[1]

        # Initialize weights
        distance_weights = np.ones(total_num_features)

        iteration = 0
        weight_history = []


        # Iterate till max iteration reached or all weights are really tiny
        while (iteration < self.max_iter):
            # Run Core Relief-based algorithm
            core_fit = core.fit(self.X_mat, self._y, distance_weights, self.weight_flag)

            #reset distance_weights so that only previous run affects next one
            distance_weights = np.ones(total_num_features)

            # When all weights become 0, break
            if all(w == 0 for w in core_fit.feature_importances_):
                break

            # Update weights
            feature_weights = core_fit.feature_importances_

            # if weight change is minimal, stop running iter and break
            # if no_diff is True, that means all features do not have a significant difference in weights between previous and current run.
            no_diff = True
            # if first iteration, set false
            if iteration == 0:
                no_diff = False
            else:
                for i in range(len(feature_weights)):
                    #previous array of feature_weights
                    prev = weight_history[len(weight_history)-1]
                    diff = abs(prev[i] - feature_weights[i])
                    # first encounter of value that has difference greater than threshold, set no_diff to False, and break out of checking loop
                    if diff >= 0.0001:
                        no_diff = False
                        break;
            if no_diff:
                break;

            mx = max(feature_weights)

            weight_history.append(feature_weights)

            #normalize and update scores
            #negative numbers -> 0, positive numbers become normalized so maximum is 1.
            for i in range(0, len(feature_weights)):
                if feature_weights[i] <= 0:
                    feature_weights[i] = 0
                else:
                    feature_weights[i] = feature_weights[i] / mx

            distance_weights += feature_weights

            iteration += 1

        self.feature_importances_ = weight_history[len(weight_history)-1]
        self.history = weight_history
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]
        return self

    #=========================================================================#

    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_select` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_select}
            Reduced feature matrix

        """

        return X[:, self.top_features_[:self.n_features_to_select]]

        # return X[:, self.top_features_]

    #=========================================================================#

    def fit_transform(self, X, y):
        # def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then reduces the feature set down to the top `n_features_to_select` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_select}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X)
