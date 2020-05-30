from sklearn.base import BaseEstimator
import copy
import numpy as np

class Iter(BaseEstimator):

    def __init__(self,relief_object,max_iter):
        '''
        :param relief_object:       Must be an object that implements the standard sklearn fit function, and after fit, has attribute feature_importances_
                                    that can be accessed. Scores must be a 1D np.ndarray of length # of features. The fit function must also be able to
                                    take in an optional 1D np.ndarray 'weights' parameter of length num_features.
        :param max_iter:            Maximum number of iterations to run
        '''

        if not self.check_is_int(max_iter) or max_iter < 0:
            raise Exception('max_iter must be a nonnegative integer')

        self.relief_object = relief_object
        self.max_iter = max_iter

    def fit(self, X, y):
        """Scikit-learn required: Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features} Training instances to compute the feature importance scores from
        y: array-like {n_samples}             Training labels

        Returns
         -------
         self
        """

        #Iterate, feeding the resulting weights of the first run into the fit of the next run (how are they translated?)
        last_iteration_scores = None
        for i in range(self.max_iter):
            copy_relief_object = copy.deepcopy(self.relief_object)
            if i == 0:
                copy_relief_object.fit(X,y)
                last_iteration_scores = copy_relief_object.feature_importances_
            else:
                copy_relief_object.fit(X,y,weights=last_iteration_scores)
                last_iteration_scores = copy_relief_object.feature_importances_

        #Save final FI as feature_importances_
        self.feature_importances_ = last_iteration_scores
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        return self

    def check_is_int(self, num):
        try:
            n = float(num)
            if num - int(num) == 0:
                return True
            else:
                return False
        except:
            return False

    def check_is_float(self, num):
        try:
            n = float(num)
            return True
        except:
            return False