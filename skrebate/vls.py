from sklearn.base import BaseEstimator

class VLS(BaseEstimator):

    def __init__(self,relief_object,num_feature_subset=40,size_feature_subset=5):
        '''
        :param relief_object:           Must be an object that implements the standard sklearn fit function, and after fit, has attribute feature_importances_
                                        that can be accessed. Scores must be a 1D np.ndarray of length # of features.
        :param num_feature_subset:      Number of feature subsets generated at random
        :param size_feature_subset:     Number of features in each subset. Cannot exceed number of features.
        '''

        self.relief_object = relief_object
        self.num_feature_subset = num_feature_subset
        self.size_feature_subset = size_feature_subset
        pass

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

        return self