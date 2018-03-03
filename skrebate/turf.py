import numpy as np
import time
import warnings
import sys
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
# from sklearn.feature_selection.base import SelectorMixin
from sklearn.externals.joblib import Parallel, delayed
# from .scoring_utils import get_row_missing, ReliefF_compute_scores
from multisurf import *
from surf import *
from surfstar import *
from relieff import *


class TuRF(BaseEstimator, TransformerMixin):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, core_algorithm, n_features_to_select=10, n_neighbors=100, step=0.1,  discrete_threshold=10, verbose=False, n_jobs=1):
        """Sets up TuRF to perform feature selection.

        Parameters
        ----------
        n_features_to_select: int (default: 10)
            the number of top features (according to the relieff score) to
            retain after feature selection is applied.
        verbose: bool (default: False)
            If True, output timing of distance array and scoring
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.
        core_algorithm: Core Relief Algorithm to perform TuRF iterations on

        """
        self.core_algorithm = core_algorithm
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.step = step
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs

    #=========================================================================#
    # headers = list(genetic_data.drop("class",axis=1))
    def fit(self,X,y,headers):
    #def fit(self, X, y):

        self.X_mat = X
        self._y = y
        self.headers = headers
        print(self.X_mat)
        if self.core_algorithm == "MultiSURF":

            num_features = X.shape[1]
            print(num_features)
            iter_count = 0
            features_iter = []
            headers_iter = []
            while(num_features > self.n_features_to_select):
                # core = MultiSURF()
                core = ReliefF()
                #core_fit = core.fit(self.X_mat,self._y, self.headers)
                core_fit = core.fit(self.X_mat, self._y)

                features_iter.append(core_fit.feature_importances_)
                headers_iter.append(self.headers)
                print(features_iter)

                if type(self.step) is float:
                    feature_rem = int(np.round(num_features*self.step))
                    # For the edge case when data has 5 or less features
                    if feature_rem == 0:
                        feature_rem = 1
                    num_features -= feature_rem
                    print(num_features)
                    print(num_features, features_iter[iter_count].argsort()[-num_features:])
                    select = np.array(features_iter[iter_count].argsort()[-num_features:])
                    print(select)
                    self.X_mat = self.X_mat[:, features_iter[iter_count].argsort()[-num_features:]]
                    print(self.X_mat)
                    print(self.X_mat.shape)
                    #self.headers = self.headers[select]
                    self.headers = [self.headers[i] for i in select]
                elif type(self.step) is int:
                    num_features -= self.step
                    self.X_mat = self.X_mat[:, features_iter[iter_count].argsort()[-num_features:]]
                    self.headers = self.headers[features_iter[iter_count].argsort()[-num_features:]]

                iter_count += 1

            # For the last iteration
            # core = MultiSURF()
            core = ReliefF()
            core_fit = core.fit(self.X_mat,self._y)
            #core_fit = core.fit(self.X_mat, self._y)
            features_iter.append(core_fit.feature_importances_)
            headers_iter.append(self.headers)

            self.num_iter = iter_count+1
            self.feature_history = list(zip(headers_iter,features_iter))
            #self.feature_history = features_iter
            print(self.feature_history)
            self.feature_importances_ = core_fit.feature_importances_
            #self.top_features_ = np.argsort(self.feature_importances_)[::-1]
            self.top_features_ = [headers.index(i) for i in s]
            return self

        elif self.core_algorithm == "SURF":
            self.surf = SURF()
            return self.surf.fit(X, y)
        elif self.core_algorithm == "SURFstar":
            self.surfstar = SURFstar()
            return self.surfstar.fit(X, y)

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
        #return X[:, self.top_features_[:self.n_features_to_select]]
        return X[:, self.top_features_]

    #=========================================================================#

    def fit_transform(self, X, y, headers):
    #def fit_transform(self, X, y):
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
        self.fit(X, y, headers)
        #self.fit(X, y)
        return self.transform(X)
