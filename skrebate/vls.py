from sklearn.base import BaseEstimator
import copy
import random
import numpy as np

class VLS(BaseEstimator):

    def __init__(self,relief_object,num_feature_subset=40,size_feature_subset=5):
        '''
        :param relief_object:           Must be an object that implements the standard sklearn fit function, and after fit, has attribute feature_importances_
                                        that can be accessed. Scores must be a 1D np.ndarray of length # of features. The fit function must also be able to
                                        take in an optional 1D np.ndarray 'weights' parameter of length num_features.
        :param num_feature_subset:      Number of feature subsets generated at random
        :param size_feature_subset:     Number of features in each subset. Cannot exceed number of features.
        '''

        if not self.check_is_int(num_feature_subset) or num_feature_subset < 0:
            raise Exception('num_feature_subset must be a nonnegative integer')

        if not self.check_is_int(size_feature_subset) or size_feature_subset < 0:
            raise Exception('size_feature_subset must be a nonnegative integer')

        self.relief_object = relief_object
        self.num_feature_subset = num_feature_subset
        self.size_feature_subset = size_feature_subset

    def fit(self, X, y,weights=None):
        """Scikit-learn required: Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features} Training instances to compute the feature importance scores from
        y: array-like {n_samples}             Training labels

        Returns
         -------
         self
        """

        #Make subsets with all the features
        num_features = X.shape[1]
        subsets = self.make_subsets(list(range(num_features)),self.num_feature_subset,self.size_feature_subset)

        #Fit each subset in parallel
        scores = []
        for subset in subsets:
            new_X = self.custom_transform(X,subset)
            copy_relief_object = copy.deepcopy(self.relief_object)
            if weights == None:
                copy_relief_object.fit(new_X,y)
            else:
                copy_relief_object.fit(new_X,y,weights=weights)
            raw_score = copy_relief_object.feature_importances_
            score = np.empty(num_features)
            score.fill(np.NINF)
            counter = 0
            for index in subset:
                score[index] = raw_score[counter]
                counter+=1
            scores.append(score)
        scores = np.array(scores)

        #Merge results by selecting largest found weight for each feature
        max_scores = []
        for score in scores.T:
            max_scores.append(np.max(score))
        max_scores = np.array(max_scores)

        #Save FI as feature_importances_
        self.feature_importances_ = max_scores
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        return self

    def custom_transform(self,X,indices_to_preserve):
        return X[:,indices_to_preserve]

    def make_subsets(self,possible_indices,num_feature_subset,size_feature_subset):
        if num_feature_subset * size_feature_subset < len(possible_indices):
            raise Exception('num_feature_subset * size_feature_subset must be >= number of total features')

        random.shuffle(possible_indices)
        remaining_indices = copy.deepcopy(possible_indices)

        subsets = []
        while True:
            subset = []
            while len(remaining_indices) > 0 and len(subset) < size_feature_subset:
                subset.append(remaining_indices.pop(0))
            subsets.append(subset)
            if len(remaining_indices) < size_feature_subset:
                break

        if len(remaining_indices) != 0:
            while len(remaining_indices) < size_feature_subset:
                index_bad = True
                while index_bad:
                    potential_index = random.choice(possible_indices)
                    if not (potential_index in remaining_indices):
                        remaining_indices.append(potential_index)
                        break
        subsets.append(remaining_indices)

        subsets_left = num_feature_subset - len(subsets)
        for i in range(subsets_left):
            subsets.append(random.sample(possible_indices,size_feature_subset))

        return subsets

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

    def transform(self, X):
        if X.shape[1] < self.relief_object.n_features_to_select:
            raise ValueError('Number of features to select is larger than the number of features in the dataset.')

        return X[:, self.top_features_[:self.relief_object.n_features_to_select]]

    def fit_transform(self, X, y, weights=None):
        self.fit(X, y, weights)
        return self.transform(X)