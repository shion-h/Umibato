#
# _plotmodule.py
#
# Copyright (c) 2020 Shion Hosoda
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#

import math
import numpy as np
import matplotlib.pyplot as plt


def plot_state(state_timepoint_df, K, color_list, title, 
               ax, markersize=10, fontsize=20):
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Timepoint', fontsize=fontsize)
    plt.sca(ax)
    plt.yticks(range(K), 
               ['State' + str(state) for state in range(K, 0, -1)],
               fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    ax.set_ylim([-1, K])
    for state in list(range(K)):
        ax.axhline(y=K - 1 - state, color=color_list[state], 
                   linestyle='-')
    this_state_list = state_timepoint_df['state'].tolist()
    timepoints = state_timepoint_df['timepoint'].tolist()
    for state, timepoint in zip(this_state_list, timepoints):
        ax.plot(timepoint, K - 1 - state, 's',
                color=color_list[state], markersize=markersize)

def plot_directed_network(intensity_df, ax, title, 
                          intensity_threshold = 0.25, 
                          scale = 20, alpha=0.5, fig_wh=1.3):
    ax.set_xlim(-fig_wh, fig_wh)
    ax.set_ylim(-fig_wh, fig_wh)
    ax.set_xticks([])
    ax.set_yticks([])
    feature_list = intensity_df.columns.tolist()
    M = len(feature_list)
    delta_theta = 2 * np.pi / M
    circle_size = 200 / M
    for i, source in enumerate(feature_list):
        source_theta = i * delta_theta
        x0 = np.sin(source_theta)
        y0 = np.cos(source_theta)
        for j, target in enumerate(feature_list):
            if j != i:
                target_theta = j * delta_theta
                x1 = np.sin(target_theta)
                y1 = np.cos(target_theta)
                delta_x = x1 - x0
                delta_y = y1 - y0
                intensity = intensity_df.loc[source, target]
                head_length = None
                # if arrow or T-shaped
                if intensity > 0:
                    head_length = 0.1
                else:
                    head_length = 1e-10
                # the length depends on if arrow or T-shaped
                margin = 0.15 + head_length
                arrow_norm = math.sqrt(delta_x**2 + delta_y**2)
                margin_rate = margin / arrow_norm
                delta_x = delta_x * (1 - margin_rate)
                delta_y = delta_y * (1 - margin_rate)
                width = abs(intensity)/scale
                if intensity > intensity_threshold:
                    ax.arrow(x0, y0, delta_x, delta_y, width=width, 
                             color='r', 
                             head_length=head_length, alpha=alpha)
                elif intensity < -intensity_threshold:
                    ax.arrow(x0, y0, delta_x, delta_y, width=width, 
                             color='b', head_length=head_length, 
                             head_width=0.1, alpha=alpha)
        ax.text(x0, y0, source, color='black', 
                size=circle_size, ha="center", va="center", 
                fontname='monospace', fontsize=25, 
                bbox = dict(boxstyle=f"circle,pad={0.1}", 
                            fc="lightgrey"))
    ax.set_title(title, fontsize=30)
