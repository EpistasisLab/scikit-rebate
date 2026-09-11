from sklearn.base import BaseEstimator
import copy
import numpy as np

class TURF(BaseEstimator):
    def __init__(self, relief_object, pct=0.5, num_scores_to_return=100):
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
        self.rank_absolute = self.relief_object.rank_absolute

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
        num_features = X.shape[1]
        self.num_scores_to_return = min(self.num_scores_to_return,num_features)

        if self.num_scores_to_return != num_features and self.pct == 1:
            raise Exception('num_scores_to_return != num_features and pct == 1. TURF will never reach your intended destination.')

        #Find out out how many features to use in each iteration
        current_features = list(range(num_features))
        eliminated_tiers = []

        N = int(1 / float(self.pct)) if self.check_is_float(self.pct) else num_features // self.pct
        for i in range(N):
            X_subset = X[:, current_features]
            relief = copy.deepcopy(self.relief_object)
            relief.fit(X_subset, y)

            importances = relief.feature_importances_
            ranks = np.argsort(importances)[::-1]
            keep_count = int(np.ceil(len(current_features) * (1 - self.pct)))
            keep_indices = sorted(ranks[:keep_count])
            eliminate_indices = [j for j in range(len(current_features)) if j not in keep_indices]

            eliminated_tiers.append([current_features[j] for j in eliminate_indices])
            current_features = [current_features[j] for j in keep_indices]

            if len(current_features) <= self.num_scores_to_return:
                break

        # Final round on the reduced set
        X_final = X[:, current_features]
        relief = copy.deepcopy(self.relief_object)
        relief.fit(X_final, y)
        final_scores = relief.feature_importances_

        # Build full-length feature_importances_ with tier scores for eliminated features
        full_scores = np.zeros(num_features)
        full_scores[current_features] = final_scores

        # Tier score fallback
        min_score = min(final_scores)
        max_score = max(final_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
        tier_penalty = 0.01 * score_range

        for k, tier in enumerate(reversed(eliminated_tiers)):
            tier_score = min_score - tier_penalty * (k + 1)
            for idx in tier:
                full_scores[idx] = tier_score

        self.feature_importances_ = full_scores
        self.top_features_ = np.argsort(np.abs(full_scores) if self.rank_absolute else full_scores)[::-1]
        return self


    def check_is_int(self, num):
        try:
            return float(num).is_integer()
        except:
            return False

    def check_is_float(self, num):
        try:
            float(num)
            return True
        except:
            return False

    def transform(self, X):
        if X.shape[1] < self.relief_object.n_features_to_select:
            raise ValueError('Number of features to select is larger than the number of features in the dataset.')

        return X[:, self.top_features_[:self.relief_object.n_features_to_select]]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)