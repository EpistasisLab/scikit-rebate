import numpy as np
import time
import warnings
import sys
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
# from sklearn.feature_selection.base import SelectorMixin
from sklearn.externals.joblib import Parallel, delayed
# from .scoring_utils import get_row_missing, ReliefF_compute_scores
from .multisurf import MultiSURF
from .multisurfstar import MultiSURFstar
from .surf import SURF
from .surfstar import SURFstar
from .relieff import ReliefF


class TuRF(BaseEstimator, TransformerMixin):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, core_algorithm, n_features_to_select=10, n_neighbors=100, pct=0.5, discrete_threshold=10, verbose=False, n_jobs=1):
        """Sets up TuRF to perform feature selection.

        Parameters
        ----------
        core_algorithm: Core Relief Algorithm to perform TuRF iterations on
        n_features_to_select: int (default: 10)
            the number of top features (according to the relieff score) to
            retain after feature selection is applied.
        pct: float/int (default: 0.5)
            If of type float, describes the fraction of features to be removed in each iteration.
            If of type int, describes the number of features to be removed in each iteration.
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
        self.pct = pct
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs

    #=========================================================================#
    # headers = list(genetic_data.drop("class",axis=1))
    def fit(self, X, y, headers):
        """
        Uses the input `core_algorithm` to determine feature importance scores at each iteration.
        At every iteration, a certain number(determined by input parameter `pct`) of least important
        features are removed, until the feature set is reduced down to the top `n_features_to_select` features.

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
        Copy of the TuRF instance
        """

        self.X_mat = X
        self._y = y
        self.headers = headers
        self._num_attributes = len(self.X_mat[0]) 
        self._lost = {}
        
        #Combine TuRF with specified 'core' Relief-based algorithm
        if self.core_algorithm.lower() == "multisurf":
            core = MultiSURF(n_features_to_select=self.n_features_to_select, discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        elif self.core_algorithm.lower() == "multisurfstar":
            core = MultiSURFstar(n_features_to_select=self.n_features_to_select, discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        elif self.core_algorithm.lower() == "surf":
            core = SURF(n_features_to_select=self.n_features_to_select, discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        elif self.core_algorithm.lower() == "surfstar":
            core = SURFstar(n_features_to_select=self.n_features_to_select, discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        elif self.core_algorithm.lower() == "relieff":
            core = ReliefF(n_features_to_select=self.n_features_to_select, n_neighbors=self.n_neighbors, discrete_threshold=self.discrete_threshold, verbose=self.verbose, n_jobs=self.n_jobs)

        num_features = X.shape[1]

        iter_count = 0
        features_iter = []
        headers_iter = []
        feature_retain_check = 0
        
        #Determine maximum number of iterations. 
        iterMax = int(1/float(self.pct))
        
        #Main iterative loop of TuRF
        while(iter_count < iterMax):
            #Run Core Relief-based algorithm
            core_fit = core.fit(self.X_mat, self._y)
            features_iter.append(core_fit.feature_importances_) #HISTORY
            headers_iter.append(self.headers) #HISTORY
            
            #Calculate features to keep
            perc_retain = 1 - self.pct
            feature_retain = int(np.round(num_features*perc_retain))
            
            # Edge case (ensures that each iteration, at least one feature is removed)
            if feature_retain == feature_retain_check:
                feature_retain -= 1

            num_features = feature_retain
            feature_retain_check = feature_retain
            #Identify the index location of the top 'num_feature' scoring features (for this particular iteration)
            select = np.array(features_iter[iter_count].argsort()[-num_features:])
            #Make index list of features not removed
            non_select = np.array(features_iter[iter_count].argsort()[:num_features])
            #Make a dictionary that stores dropped features and the iteration they were dropped.
            for i in non_select:
                self._lost[self.headers[i]] = iterMax - iter_count #For feature name, store iteration rank it was removed (bigger rank for sooner removal)
                
            #Drop non-selected features and headers. 
            self.X_mat = self.X_mat[:, select] #select all instances and only features indexed from select. 
            self.headers = [self.headers[i] for i in select]

            iter_count += 1

        #Final scoring iteration
        core_fit = core.fit(self.X_mat, self._y)
        features_iter.append(core_fit.feature_importances_) #HISTORY
        headers_iter.append(self.headers) #HISTORY
        iter_count += 1

        self.num_iter = iter_count
        self.feature_history = list(zip(headers_iter, features_iter)) #HISTORY

        #Prepare for assigning token scores to features that had been removed in a previous TuRF iteration.  These scores are only meaningful in that they give an idea of when these feature(s) were removed. 
        low_score = min(core_fit.feature_importances_)
        reduction = 0.01 * (max(core_fit.feature_importances_) - low_score)

        #For consistency we report feature importances ordered in same way as original dataset.  Same is true for headers. 
        #Step through each feature name
        self.feature_importances_= []

        for i in headers_iter[0]:
            #Check lost dictionary
            if i in self._lost:
                self.feature_importances_.append(low_score - reduction * self._lost[i]) #append discounted score as a marker of when the feature was removed. 
            else: #Feature made final cut
                score_index = self.headers.index(i)
                self.feature_importances_.append(core_fit.feature_importances_[score_index])

        #Turn feature imporance list into array
        self.feature_importances_= np.array(self.feature_importances_)
        #self.feature_importances_ = core_fit.feature_importances_
        
        self.top_features_ = [headers.index(i) for i in self.headers]
        self.top_features_ = self.top_features_[::-1]
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
        if self._num_attributes < self.n_features_to_select:
            raise ValueError('Number of features to select is larger than the number of features in the dataset.')
        
        return X[:, self.top_features_[:self.n_features_to_select]]
        #return X[:, self.top_features_]

    #=========================================================================#

    def fit_transform(self, X, y, headers):
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
        self.fit(X, y, headers)
        return self.transform(X)
