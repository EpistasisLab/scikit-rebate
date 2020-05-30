from sklearn.base import BaseEstimator
import copy
import numpy as np

class TURF(BaseEstimator):

    def __init__(self,relief_object,pct=0.5,num_scores_to_return=100):
        '''
        :param relief_object:           Must be an object that implements the standard sklearn fit function, and after fit, has attributes feature_importances_
                                        and top_features_ that can be accessed. Scores must be a 1D np.ndarray of length # of features.
        :param pct:                     % of features to remove from removing features each iteration (if float). Or # of features to remove each iteration (if int)
        :param num_scores_to_return:    Number of nonzero scores to return after training. Default = min(num_features, 100)
        '''
        if not self.check_is_int(num_scores_to_return) or num_scores_to_return < 0:
            raise Exception('num_scores_to_return must be a nonnegative integer')

        if (not self.check_is_int(pct) and not self.check_is_float(pct)) or pct < 0:
            raise Exception('pct must be a nonnegative integer/float')

        if (not self.check_is_int(pct) and self.check_is_float(pct)) and (pct < 0 or pct > 1):
            raise Exception('if pct is a float, it must be from [0,1]')

        self.relief_object = relief_object
        self.pct = pct
        self.num_scores_to_return = num_scores_to_return

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
        #Adjust num_scores_to_return
        num_features = X.shape[0]
        self.num_scores_to_return = min(self.num_scores_to_return,num_features)

        if self.num_scores_to_return != num_features and self.pct == 1:
            raise Exception('num_scores_to_return != num_features and pct == 1. TURF will never reach your intended destination.')

        #Find out out how many features to use in each iteration
        features_per_iteration = self.get_features_per_iteration(num_features,self.pct,self.num_scores_to_return)

        #Iterate runs
        binary_scores_existence_tracker = np.ones(num_features) #1 means score still left

        copy_relief_object = copy.deepcopy(self.relief_object)
        copy_relief_object.fit(X, y)
        features_per_iteration.pop(0)
        for num_features_to_use_in_iteration in features_per_iteration:
            #Find top raw features indices
            best_raw_indices = copy_relief_object.top_features_[:num_features_to_use_in_iteration]

            #Map raw features indices to original feature indices array
            onesCounter = 0
            for i in range(len(binary_scores_existence_tracker)):
                if not (onesCounter in best_raw_indices):
                    binary_scores_existence_tracker[i] = 0
                if binary_scores_existence_tracker[i] == 1:
                    onesCounter+=1

            #Get new X
            new_indices = []
            for i in range(len(binary_scores_existence_tracker)):
                if binary_scores_existence_tracker[i] == 1:
                    new_indices.append(i)

            new_X = X[:,new_indices]

            #fit
            copy_relief_object = copy.deepcopy(self.relief_object)
            copy_relief_object.fit(new_X, y)

        #Return remaining scores in their original indices, having zeros for the rest
        raw_scores = copy_relief_object.feature_importances_
        counter = 0
        for i in range(len(binary_scores_existence_tracker)):
            if binary_scores_existence_tracker[i] == 1:
                binary_scores_existence_tracker[i] = raw_scores[counter]
                counter += 1

        # Save FI as feature_importances_
        self.feature_importances_ = binary_scores_existence_tracker
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        return self

    def get_features_per_iteration(self,num_features,pct,num_scores_to_return):
        features_per_iteration = [num_features]
        features_left = num_features
        if num_features != num_scores_to_return:
            if self.check_is_int(pct) and not self.check_is_float(pct):  # Is int
                while True:
                    if features_left - num_features > num_scores_to_return:
                        features_left -= num_features
                        features_per_iteration.append(features_left)
                    else:
                        features_per_iteration.append(num_scores_to_return)
                        break
            else:  # Is float
                while True:
                    if int(features_left * pct) > num_scores_to_return:
                        features_left = int(features_left * pct)
                        features_per_iteration.append(features_left)
                    else:
                        features_per_iteration.append(num_scores_to_return)
                        break
        return features_per_iteration

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