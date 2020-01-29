import numpy as np
import pandas as pd
import time
import warnings
import sys
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
# from sklearn.feature_selection.base import SelectorMixin
#from sklearn.externals.joblib import Parallel, delayed
#import joblib
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
from .vlsrelief import VLSRelief


class VLSIter(BaseEstimator, TransformerMixin):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, core_algorithm, weight_flag=1, n_features_to_select=2, n_neighbors=100, max_iter=2, num_feature_subset=40, size_feature_subset=5, discrete_threshold=10, verbose=False, n_jobs=1):
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
        #################Default above is 2
        max_iter: int (default 10)
            the maximum number of iterations to be performed.
        num_feature_subset: int (default: 40)
            Number of subsets generated at random
        size_feature_subset: int (default 5)
            Number of features in each subset generated
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
        self.num_feature_subset = num_feature_subset
        self.size_feature_subset = size_feature_subset

    #=========================================================================#
    def fit(self, X, y, headers):
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
        headers: array-like {n_features}
            Feature names
        Returns
        -------
        Copy of the IterRelief instance
        """

        self.X_mat = X
        self._y = y
        self.headers = headers

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

        # init for vls
        num_features = self.size_feature_subset

        # Iterate till max iteration reached or all weights are really tiny
        while ((iteration < self.max_iter)):
            #init for vls
            features_scores_iter = []
            headers_iter = []
            features_selected = []

            # Run vlsRelief
            for i in range(self.num_feature_subset):
                if i == 1:
                    # grab the previous features_selected
                    # grab new features so that there is no overlap
                    features_selected_id = []
                    prev_selection = np.sort(features_selected[0])
                    index = 0
                    for i in range(total_num_features):
                        if index >= num_features:
                            features_selected_id.append(i)
                        elif prev_selection[index] != i:
                            features_selected_id.append(i)
                        else:
                            index += 1
                else:
                    features_selected_id = np.random.choice(
                        range(total_num_features), num_features, replace=False)
                    self.X_train = self.X_mat[:, features_selected_id]

                distance_weights_selected = []
                for i in features_selected_id:
                    distance_weights_selected.append(distance_weights[i])

                #run vls while taking distance weights into account
                core_fit = core.fit(self.X_train, self._y, distance_weights_selected, self.weight_flag)

                features_scores_iter.append(core_fit.feature_importances_)
                features_selected.append(features_selected_id)


            self.features_scores_iter = features_scores_iter
            self.features_selected = features_selected

            zip_feat_score = [list(zip(features_selected[i], features_scores_iter[i]))
                              for i in range(len(features_selected))]
            feat_score = sorted([item for sublist in zip_feat_score for item in sublist])
            feat_score_df = pd.DataFrame(feat_score)
            feat_score_df.columns = ['feature', 'score']
            feat_score_df = feat_score_df.groupby('feature').max().reset_index()

            feature_scores = feat_score_df.values
            feature_scores = [[int(i[0]), i[1]] for i in feature_scores]

            self.feat_score = feature_scores

            #reset distance_weights so that only previous run affects next one
            distance_weights = np.ones(total_num_features)

            # When all weights become 0, break
            feature_column, feature_weights = zip(*feature_scores)
            feature_column, feature_weights = np.asarray(feature_column), np.asarray(feature_weights)
            #print("feature_column:")
            #print(feature_column)
            #print("feature_weights")
            #print(feature_weights)
            
            if all(w == 0 for w in feature_weights):
                break

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

            # calculations for weight updates           
            mx = max(feature_weights)
            mn = min(feature_weights)
            rg = mx - mn

            weight_history.append(feature_weights)


            #normalize and update scores
            for i in range(0, len(feature_weights)):
            	if feature_weights[i] <= 0:
            		feature_weights[i] = 0
            	else:
            		feature_weights[i] = feature_weights[i] / mx
            #feature_weights = [(x - mn)/(rg) for x in feature_weights]

            index = 0
            for i in feature_column:
                distance_weights[i] = feature_weights[index]
                index+=1


            iteration += 1

        self.feature_importances_ = weight_history[len(weight_history)-1]
        self.history = weight_history
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]
        #print("feature_importances:")
        #print(self.feature_importances_)
        #print("top_features_")
        #print(self.top_features_)

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
