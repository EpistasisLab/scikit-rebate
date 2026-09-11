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
from .surfstar import SURFstar
from joblib import Parallel, delayed
from .scoring_utils import MultiSURF_compute_scores
# import matplotlib.pyplot as plt
import time
# from pyinstrument import Profiler
# from line_profiler import LineProfiler
from .scoring_utils import compute_score, ramp_vec


class MultiSURF(SURFstar):

    """Feature selection using data-mined expert knowledge.
    Based on the MultiSURF algorithm as introduced in:
    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.
    """

############################# MultiSURF ########################################
    def _find_neighbors(self, inst):
        """ Identify nearest hits and misses within radius defined by average distance and standard deviation around each target training instance.
        This works the same regardless of endpoint type. """
        dist_vect = []
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                dist_vect.append(self._distance_array[locator[0]][locator[1]])

        dist_vect = np.array(dist_vect)
        
        if self.distarray_has_nan:
            inst_avg_dist = np.nanmean(dist_vect)
            true_std = np.nanstd(dist_vect)
            inst_deadband = true_std / 2.
            # Defining a narrower radius based on the average instance distance minus 1/2 the standard deviation of instance distances.
            near_threshold = inst_avg_dist - inst_deadband
        else:
            inst_avg_dist = np.mean(dist_vect)
            true_std = np.std(dist_vect)
            inst_deadband = true_std / 2.
            # Defining a narrower radius based on the average instance distance minus 1/2 the standard deviation of instance distances.
            near_threshold = inst_avg_dist - inst_deadband


        # NEW: unique mean, std, and deadband values for this target instance, used to construct the expected curve
        # self.instance_dist_stats.append((inst_avg_dist, true_std, inst_deadband))

        NN_near = []
        for j in range(self._datalen):
            if inst != j:
                locator = [inst, j]
                if inst < j:
                    locator.reverse()
                d = self._distance_array[locator[0]][locator[1]]
                # NEW: how many standard deviations d is from the current mean distance
                # std_d = (d - inst_avg_dist) / true_std
                if d < near_threshold:
                    NN_near.append(j)
                #     # NEW: for plotting
                #     self.distance_weight_log.append((d, 1.0))
                #     self.std_weight_log.append((std_d, 1.0))
                # else:
                #     self.distance_weight_log.append((d, 0.0))
                #     self.std_weight_log.append((std_d, 0.0))

        return np.array(NN_near)

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield MultiSURF scores. """
        nan_entries = np.isnan(self._X)

        # self.distance_weight_log = []  # Reset log before run
        # self.std_weight_log = []
        # self.instance_dist_stats = []

        # lp = LineProfiler()
        # lp.add_function(compute_score)
        # lp.add_function(ramp_vec)
        # profiler = Profiler()
        # profiler.start()
        NNlist = [self._find_neighbors(datalen) for datalen in range(self._datalen)]

        # lp.enable()
        
        if isinstance(self._weights, np.ndarray) and self.weight_final_scores:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                MultiSURF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                          NN_near, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type, self._weights)
                                                        for instance_num, NN_near in zip(range(self._datalen), NNlist)), axis=0)
        else:
            scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
                MultiSURF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                          NN_near, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
                                                        for instance_num, NN_near in zip(range(self._datalen), NNlist)), axis=0)

        # lp.disable()
        # lp.print_stats()
        # profiler.stop()
        # profiler.print()
        
        # print(scores)
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

    #         y_vals = []
    #         for x in x_vals:
    #             if x < (mean_dist - dead_band):
    #                 y_vals.append(1.0)
    #             else:
    #                 y_vals.append(0.0)

    #         if y_vals is not None:
    #             # plt.plot(x_vals, y_vals, label='Expected', linewidth=2, color='black')
    #             # NEW: use x_vals STD instead
    #             plt.plot(x_vals_std, y_vals, label='Expected', linewidth=2, color='black')

    #     plt.title(f'Distance-to-Weight Mapping: MultiSURF')
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
