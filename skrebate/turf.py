from sklearn.base import BaseEstimator

class TURF(BaseEstimator):

    def __init__(self,relief_object,pct=0.5,num_scores_to_return=None):
        '''
        :param relief_object:           Must be an object that implements the standard sklearn fit function, and after fit, has attribute feature_importances_
                                        that can be accessed. Scores must be a 1D np.ndarray of length # of features.
        :param pct:                     % of features to remove from removing features each iteration (if float). Or # of features to remove each iteration (if int)
        :param num_scores_to_return:    Number of nonzero scores to return after training. Default = min(num_features, 100)
        '''

        self.relief_object = relief_object
        self.pct = pct
        self.num_scores_to_return = num_scores_to_return
        pass