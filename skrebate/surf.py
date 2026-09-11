# -*- coding: utf-8 -*-

"""
scikit-rebate was primarily developed at the University of Pennsylvania and at the Cedars-Sinai Health Sciences University by:
    - Ryan J. Urbanowicz (ryanurb@upenn.edu)
    - Randal S. Olson (rso@randalolson.com)
    - Pete Schmitt (pschmitt@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Ting-Hui Wu (tinghui333w@gmail.com)
    - Kia Kazemi-Nia (kia.kazemi-nia@cshs.org)
    - and many more generous open source contributors
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import numpy as np
from joblib import Parallel, delayed
from .relieff import ReliefF
from .scoring_utils import SURF_compute_scores
# import matplotlib.pyplot as plt


class SURF(ReliefF):

    """Feature selection using data-mined expert knowledge.
    Based on the SURF algorithm as introduced in:
    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.
    """

    def __init__(self, n_features_to_select=10, categorical_features=None, 
                 categorical_threshold=10, multiclass_threshold=10, verbose=False, n_jobs=1, weight_final_scores=False,
                 rank_absolute=False, label_type=None):
        """Sets up SURF to perform feature selection.
        Parameters
        ----------
        n_features_to_select: int (default: 10)
            the number of top features (according to the feature importance score) to
            retain after feature selection is applied.
        categorical_features: list (default: None)
            list of index columns indicating features to be treated as categorical.
            Any features not explicitly listed will be treated as quantitative by default.
        categorical_threshold: int (default: 10)
            Value used to determine if a feature is categorical/discrete or continuous.
            If the number of unique values in a feature is > categorical_threshold, then it is
            considered continuous, or categorical otherwise.
        multiclass_threshold: int (default: 10)
            Value used to determine if a target is multiclass or continuous.
            If the number of unique values in the target variable is > multiclass_threshold, then it is
            considered continuous. If it is <= multiclass_threshold and > 2, it is considered multiclass.
        verbose: bool (default: False)
            If True, output the time taken for the distance array computation and scoring.
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.
        weight_final_scores: bool (default: False)
            Whether to multiply given weights (in fit) to final scores. Only applicable if weights are given.
        rank_absolute: bool (default: False)
            Whether to rank features according to the absolute value of their feature importance score.
        label_type: str (default: None)
            The default value is None, in which case the function automatically infers the label type
            based on the number of unique labels: 2 for 'binary', 3-10 for 'multiclass', and >10 for 'continuous'.
            Alternatively, you can specify one of the following strings: 'binary', 'multiclass', or 'continuous'.
        """
        self.n_features_to_select = n_features_to_select
        self.categorical_features = categorical_features
        self.categorical_threshold = categorical_threshold
        self.multiclass_threshold = multiclass_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.weight_final_scores = weight_final_scores
        self.rank_absolute = rank_absolute
        self.label_type = label_type

        # NEW: for logging (distance, weight) pairs, logging (std, weight) pairs, and approximating expected curve for MultiSURF
        # self.distance_weight_log = []
        # self.std_weight_log = []
        # self.instance_dist_stats = []

############################# SURF ############################################
    def _find_neighbors(self, inst, avg_dist):
        """ Identify nearest hits and misses within radius defined by average distance over whole distance array.
        This works the same regardless of endpoint type. """
        NN = []
        min_indicies = []

        # # NEW: STD calculated from distance array, for use in plot_distance_weight_map
        # dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
        # std_dist = dists_flat.std()

        for i in range(self._datalen):
            if inst != i:
                locator = [inst, i]
                if i > inst:
                    locator.reverse()
                d = self._distance_array[locator[0]][locator[1]]
                # # NEW: how many std's d is away from the mean distance
                # std_d = (d - avg_dist) / std_dist
                if d < avg_dist:  # Defining the neighborhood with an average distance radius.
                    min_indicies.append(i)
                #     # NEW: for plotting
                #     self.distance_weight_log.append((d, 1.0))
                #     self.std_weight_log.append((std_d, 1.0))
                # else:
                #     self.distance_weight_log.append((d, 0.0))
                #     self.std_weight_log.append((std_d, 0.0))
        for i in range(len(min_indicies)):
            NN.append(min_indicies[i])
        return np.array(NN, dtype=np.int32)

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield SURF scores. """
        # using numpy to compute global mean
        dists_flat = np.concatenate([np.array(row) for row in self._distance_array])
        
        if self.distarray_has_nan:
            avg_dist = np.nanmean(dists_flat)
        else:
            avg_dist = dists_flat.mean()

        nan_entries = np.isnan(self._X)

        # self.distance_weight_log = []  # Reset log before run
        # self.std_weight_log = []

        NNlist = [self._find_neighbors(datalen, avg_dist) for datalen in range(self._datalen)]
        if isinstance(self._weights, np.ndarray) and self.weight_final_scores:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                SURF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                     NN, self._headers, self._class_type, self._X, self._y, self._labels_std,self.data_type, self._weights)
                                                         for instance_num, NN in zip(range(self._datalen), NNlist)),axis=0)
        else:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                SURF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                     NN, self._headers, self._class_type, self._X, self._y, self._labels_std,self.data_type)
                                                         for instance_num, NN in zip(range(self._datalen), NNlist)),axis=0)

        return np.array(scores)
    
    # def plot_distance_weight_map(self, save_fig=None, show_expected=True):
    #     """Visualize actual (distance, weight) pairs collected during Relief run."""
    #     if not self.distance_weight_log:
    #         print("No data logged yet. Run the algorithm first.")
    #         return

    #     distances, weights = zip(*self.distance_weight_log)
    #     # NEW: use self.std_weight_log for plotting
    #     distances_std, weights_std = zip(*self.std_weight_log)
    #     plt.figure(figsize=(10, 6))
    #     # plt.scatter(distances, weights, alpha=0.3, s=10, label='Observed')
    #     plt.scatter(distances_std, weights_std, alpha=0.3, s=10, label='Observed')

    #     if show_expected:
    #         x_vals = np.linspace(min(distances), max(distances), 500)
    #         # mean_dist = np.mean(distances)
    #         # std_dist = np.std(distances)
    #         # mean_dist = np.nanmean(distances)
    #         # std_dist = np.nanstd(distances)
    #         if self.distarray_has_nan:
    #             mean_dist = np.nanmean(distances)
    #             std_dist = np.nanstd(distances)
    #         else:
    #             mean_dist = np.mean(distances)
    #             std_dist = np.std(distances)

    #         # NEW: for plotting in terms of STD
    #         x_vals_std = (x_vals - mean_dist) / std_dist

    #         y_vals = []
    #         for x in x_vals:
    #             if x < mean_dist:
    #                 y_vals.append(1.0)
    #             else:
    #                 y_vals.append(0.0)

    #         if y_vals is not None:
    #             # plt.plot(x_vals, y_vals, label='Expected', linewidth=2, color='black')
    #             # NEW: use x_vals STD instead
    #             plt.plot(x_vals_std, y_vals, label='Expected', linewidth=2, color='black')

    #     plt.title(f'Distance-to-Weight Mapping: SURF')
    #     plt.xlabel('Distance from Target Instance')
    #     # plt.xlabel('Standard Deviations (Distance) from Target Instance')
    #     plt.ylabel('Scoring Weight')
    #     plt.grid(True)
    #     # NEW: grid lines different from x-tick labels:
    #     plt.gca().set_xticks(np.arange(-3, 4, 1), minor=False)
    #     plt.ylim(-1.1, 1.1)
    #     # NEW: xlim to set x-axis values between 0 and 1.0 for all graphs (consistent)
    #     # plt.xlim(0, 1.0)
    #     plt.xlim(-3.0, 3.0)
    #     # plt.xticks(np.linspace(0, 1.0, num=6))
    #     # plt.xticks([-3, -2, -1, 0, 1, 2, 3])
    #     # plt.xticks(
    #     #     [-0.5, 0, 0.5],
    #     #     ['(μ - σ/2)', 'μ', '(μ + σ/2)']
    #     # )
    #     plt.xticks(
    #         [0],
    #         ['μ']
    #     )
    #     # NEW: dotted lines for deadband zone boundaries (0.5 SD on either side of mean)
    #     # plt.axvline(x=-0.5, color='red', linestyle='dotted')
    #     # plt.axvline(x=0.5,  color='red', linestyle='dotted')
    #     plt.legend()
    #     if save_fig:
    #         plt.savefig(save_fig)
    #     else:
    #         plt.show()
