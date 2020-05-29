from sklearn.base import BaseEstimator

class Iter(BaseEstimator):

    def __init__(self,relief_object,max_iter):
        '''
        :param relief_object:       Must be an object that implements the standard sklearn fit function, and after fit, has attribute feature_importances_
                                    that can be accessed. Scores must be a 1D np.ndarray of length # of features. The fit function must also be able to
                                    take in an optional 1D np.ndarray 'weights' parameter of length num_features.
        :param max_iter:            Maximum number of iterations to run
        '''
        self.relief_object = relief_object
        self.max_iter = max_iter
        pass