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
from .relieff import ReliefF
from joblib import Parallel, delayed
from .scoring_utils import ReliefF_compute_scores
# import matplotlib.pyplot as plt
import sys


class MuRelief(ReliefF):

    """Feature selection using data-mined expert knowledge.
    Based on the μ-Relief algorithm as introduced in:
    Aggarwal, N., Shukla, U., Saxena, G.J. et al.
    Mean based relief: An improved feature selection method based on ReliefF.
    """

############################# μ-Relief ########################################
    def _find_neighbors(self, inst):
        """ Identify nearest hits and misses, i.e. the k largest absolute differences between: 
        1) distance from target instance and neighboring instance 2) mean distance from target instance.
        """
        dist_vect = []
        # unique means per class/outcome group: key = class, value = (list(distances), list(indices))
        means_calc = {}
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                dist_vect.append(self._distance_array[locator[0]][locator[1]])
                d = self._distance_array[locator[0]][locator[1]]
                
                # calculating mean for each class/outcome group
                if self._class_type in ('binary', 'multiclass'):
                    label = self._y[j]

                    # If key doesn't exist: create ([], [])
                    if label not in means_calc:
                        means_calc[label] = ([], [])

                    # Unpack lists
                    dist_list, idx_list = means_calc[label]

                    # Append data
                    dist_list.append(d)
                    idx_list.append(j)
                
                elif self._class_type == 'continuous':
                    # Determine group 1 or 0
                    if abs(self._y[inst] - self._y[j]) < self._labels_std:
                        group = 1
                    else:
                        group = 0

                    # If key doesn't exist: create ([], [])
                    if group not in means_calc:
                        means_calc[group] = ([], [])

                    # Unpack lists
                    dist_list, idx_list = means_calc[group]

                    # Append data
                    dist_list.append(d)
                    idx_list.append(j)

        dist_vect = np.array(dist_vect)
        if self.distarray_has_nan:
            inst_mean_dist = np.nanmean(dist_vect)
            true_std = np.nanstd(dist_vect)
        else:
            inst_mean_dist = np.mean(dist_vect)
            true_std = np.std(dist_vect)
        inst_deadband = true_std / 2.

        # replacing the value for the target instance to the mean so that it is never selected as its own neighbor in μ-Relief
        dist_vect = np.insert(dist_vect, inst, inst_mean_dist)

        # unique mean, std, and deadband values for this target instance, used to construct the expected curve in distance-weight plot
        # self.instance_dist_stats.append((inst_mean_dist, true_std, inst_deadband))

        # initializing all weights to 0 before the k neighbors (and only those neighbors) get weights of 1 (for plotting purposes)
        # self.distance_weight_log = [(d, 0.0) for d in dist_vect]
        # self.std_weight_log = [((d - inst_mean_dist) / true_std, 0.0) for d in dist_vect]

        # calculating mean distance for each outcome group
        for label, (dist_list, idx_list) in means_calc.items():
            if self.distarray_has_nan:
                mean_dist = float(np.nanmean(dist_list)) if dist_list else 0.0
            else:
                mean_dist = float(np.mean(dist_list)) if dist_list else 0.0
            means_calc[label] = (dist_list, idx_list, mean_dist)

        # creating a new 'absolute difference between the observed distance and the mean distance' vector
        # * mean is unique per outcome group
        abs_diff_vect = np.zeros(self._datalen)
        for _, (dist_list, idx_list, mean_dist) in means_calc.items():
            idx_arr = np.array(idx_list)
            abs_diff_vect[idx_arr] = np.abs(dist_vect[idx_arr] - mean_dist)

        # Identify neighbors-------------------------------------------------------
        """ Neighbors for Binary Endpoints: """
        if self._class_type == 'binary':
            n_list = []
            match_count = 0
            miss_count = 0
            for n_index in np.argsort(abs_diff_vect)[::-1]:
                if self._y[inst] == self._y[n_index]:  # Hit neighbor identified
                    if match_count >= self.n_neighbors:
                        continue
                    n_list.append(n_index)
                    match_count += 1
                else:  # Miss neighbor identified
                    if miss_count >= self.n_neighbors:
                        continue
                    n_list.append(n_index)
                    miss_count += 1
                
                # for distance-weight plot purposes
                # d = dist_vect[n_index]
                # self.distance_weight_log.append((d, 1.0))
                # std_d = (d - inst_mean_dist) / true_std
                # self.std_weight_log.append((std_d, 1.0))

                if match_count >= self.n_neighbors and miss_count >= self.n_neighbors:
                    break

        elif self._class_type == 'multiclass':
            n_list = []
            match_count = 0
            miss_count = dict.fromkeys(self._label_list, 0)
            for n_index in np.argsort(abs_diff_vect)[::-1]:
                if self._y[inst] == self._y[n_index]:  # Hit neighbor identified
                    if match_count >= self.n_neighbors:
                        continue
                    n_list.append(n_index)
                    match_count += 1
                else:
                    for label in self._label_list:
                        if self._y[n_index] == label:
                            if miss_count[label] >= self.n_neighbors:
                                continue
                            n_list.append(n_index)
                            miss_count[label] += 1
                
                # for distance-weight plot purposes
                # d = dist_vect[n_index]
                # self.distance_weight_log.append((d, 1.0))
                # std_d = (d - inst_mean_dist) / true_std
                # self.std_weight_log.append((std_d, 1.0))

                if match_count >= self.n_neighbors and all(v >= self.n_neighbors for v in miss_count.values()):
                    break
        else:
            n_list = []
            match_count = 0
            miss_count = 0
            for n_index in np.argsort(abs_diff_vect)[::-1]:
                if abs(self._y[inst]-self._y[n_index]) < self._labels_std:  # Hit neighbor identified
                    if match_count >= self.n_neighbors:
                        continue
                    n_list.append(n_index)
                    match_count += 1
                else:  # Miss neighbor identified
                    if miss_count >= self.n_neighbors:
                        continue
                    n_list.append(n_index)
                    miss_count += 1
                
                # for distance-weight plot purposes
                # d = dist_vect[n_index]
                # self.distance_weight_log.append((d, 1.0))
                # std_d = (d - inst_mean_dist) / true_std
                # self.std_weight_log.append((std_d, 1.0))

                if match_count >= self.n_neighbors and miss_count >= self.n_neighbors:
                    break
        
        # for distance-weight plot purposes
        # n_set = set(n_list)
        # for i in range(len(dist_vect)):
        #     if i not in n_set:  # skip neighbors
        #         d = dist_vect[i]
        #         self.distance_weight_log.append((d, 0.0))
        #         std_d = (d - inst_mean_dist) / true_std
        #         self.std_weight_log.append((std_d, 0.0))
        
        return np.array(n_list)

    def _run_algorithm(self):
        """ Runs neighbor identification and feature scoring to yield μ-Relief scores. """
        nan_entries = np.isnan(self._X)

        # self.distance_weight_log = []  # Reset log before run
        # self.std_weight_log = []
        # self.instance_dist_stats = []

        Nlist = [self._find_neighbors(datalen) for datalen in range(self._datalen)]

        if isinstance(self._weights, np.ndarray) and self.weight_final_scores:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                ReliefF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                          neighbors, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type, self._weights)
                                                        for instance_num, neighbors in zip(range(self._datalen), Nlist)), axis=0)
        else:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                ReliefF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                          neighbors, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
                                                        for instance_num, neighbors in zip(range(self._datalen), Nlist)), axis=0)

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
    #         # average per-instance mean/std
    #         x_vals = np.linspace(min(distances), max(distances), 500)
    #         means, stds, deadband = zip(*self.instance_dist_stats)
    #         mean_dist = np.mean(means)
    #         std_dist = np.mean(stds)
    #         dead_band = np.mean(deadband)
    #         # dead_band = std_dist / 4.0 if 'TBD' in self.name else 0

    #         # NEW: for plotting in terms of STD
    #         x_vals_std = (x_vals - mean_dist) / std_dist

    #         x_vals_abs_diff = np.abs(x_vals - mean_dist)

    #         # y_vals = []
    #         y_vals = np.zeros(500)
    #         neighbor_count = 0
    #         # for x in x_vals:
    #         for x_index in np.argsort(x_vals_abs_diff)[::-1]:
    #             # if x < (mean_dist - dead_band):
    #             #     y_vals.append(1.0)
    #             # else:
    #             #     y_vals.append(0.0)
    #             y_vals[x_index] = 1.0
    #             neighbor_count += 1
    #             if neighbor_count >= self.n_neighbors:
    #                 break

    #         if y_vals is not None:
    #             # plt.plot(x_vals, y_vals, label='Expected', linewidth=2, color='black')
    #             # NEW: use x_vals STD instead
    #             plt.plot(x_vals_std, y_vals, label='Expected', linewidth=2, color='black')
        
    #     # sorted_log = sorted(self.std_weight_log, key=lambda x: x[0], reverse=True)
    #     # # Write to text file
    #     # with open(save_file, "w") as f:
    #     #     for tup in sorted_log:
    #     #         f.write(f"{tup}\n")

    #     plt.title(f'Distance-to-Weight Mapping: μ-Relief')
    #     plt.xlabel('Distance from Target Instance')
    #     # plt.xlabel('Standard Deviations (Distance) from Target Instance')
    #     plt.ylabel('Scoring Weight')
    #     plt.grid(True)
    #     # NEW: grid lines different from x-tick labels:
    #     # plt.gca().set_xticks(np.arange(-3, 4, 1), minor=False)
    #     plt.ylim(-1.1, 1.1)
    #     # NEW: xlim to set x-axis values between 0 and 1.0 for all graphs (consistent)
    #     # plt.xlim(0, 1.0)
    #     # plt.xlim(-3.0, 3.0)
    #     # plt.xticks(np.linspace(0, 1.0, num=6))
    #     # plt.xticks([-3, -2, -1, 0, 1, 2, 3])
    #     plt.xticks(
    #         [-0.5, 0, 0.5],
    #         ['(μ - σ/2)', 'μ', '(μ + σ/2)']
    #     )
    #     # NEW: dotted lines for deadband zone boundaries (0.5 SD on either side of mean)
    #     plt.axvline(x=-0.5, color='red', linestyle='dotted')
    #     plt.axvline(x=0.5,  color='red', linestyle='dotted')
    #     plt.legend()
    #     if save_fig:
    #         plt.savefig(save_fig)
    #     else:
    #         plt.show()