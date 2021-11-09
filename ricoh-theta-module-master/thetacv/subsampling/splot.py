#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2016 Evripidis Gkanias, Theodoros Stouraitis
#                   <ev.gkanias@gmail.com>, <stoutheo@gmail.com>
#
#                   University of Edinburgh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ---------------------------------------------------------------------------- #
# Plot methods for the subsampling models.
# This is used for the visualisation of the topology of the ommatidia.
#

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import os, yaml

# get path of the script
cpath = os.path.dirname(os.path.abspath(__file__)) + '/'


# load parameters
with open(cpath + '/params.yaml', 'rb') as f:
    params = yaml.safe_load(f)
    sparams = params['subsampling']
    pparams = params['plot']


Limit = pparams['axis-limit']

# 2D plots


def side_slice(p_eye_x, p_eye_y, frsep=True, lines=True, radius=1.):
    """
    Plots the slice that separates the left and right eye.

    :param p_eye_x:   the horizontal coordinates
    :param p_eye_y:   the vertical coordinates
    :param frsep:     whether we want different colours for the frontal and rear areas
    :param lines:     whether we want to plot the lines that connect the frontal
                      and the rear areas (only for the dorsal area)
    """

    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')

    if frsep:
        # split frontal from rear areas
        f_eye_x, f_eye_y = p_eye_x[p_eye_x >= 0], p_eye_y[p_eye_x >= 0]
        r_eye_x, r_eye_y = p_eye_x[p_eye_x <  0], p_eye_y[p_eye_x <  0]

        ax.plot(f_eye_x, f_eye_y, 'r.', label='Frontal')
        ax.plot(r_eye_x, r_eye_y, 'y.', label='Rear')
    else:
        ax.plot(p_eye_x, p_eye_y, 'r.')

    if lines:
        # get only the points which have positive z value (dorsal)
        d_eye_x = p_eye_x[p_eye_y>=0]
        d_eye_y = p_eye_y[p_eye_y>=0]
        for i, j in zip(range(0, len(d_eye_x)/2),
                        range(len(d_eye_y)-1, len(d_eye_y)/2, -1)):
            ax.plot((d_eye_x[i], d_eye_x[j]),(d_eye_y[i], d_eye_y[j]),'k-')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    limit = Limit * radius
    plt.xlabel('Depth')
    plt.ylabel('Height')
    plt.title('Side slice')
    plt.axis([-limit, limit, -limit, limit])
    plt.show()


def top_view(all_hor_slices, line=True, radius=1.):
    """
    Plots the top view of the eye.

    :param all_hor_slices:    the horizontal lines information
    :param line:              whether we want to plot the line that separates the
                              two hemispheres
    """

    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')

    odd = False
    for h_eye_x, h_eye_y in all_hor_slices:
        if odd:
            ax.plot(h_eye_x, h_eye_y, 'r.', label='Odd')
        else:
            ax.plot(h_eye_x, h_eye_y, 'b.', label='Even')
        odd = not odd

    if line:
        ax.plot(np.zeros(2), [-Limit, Limit], 'k-')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2])
    limit = Limit * radius
    plt.xlabel('Width')
    plt.ylabel('Depth')
    plt.title('Top slice')
    plt.axis([-limit, limit, -limit, limit])
    plt.show()


def twin_hemishpere_gauss(lfeats, rfeats, radius=1., scale=0.02):
    """
    Plots the two hemispheres of the eye separately visualising the contour of
    the gaussian for every ommatidium.

    :param lfeats:    the features of the left hemisphere
    :param rfeats:    the features of the right hemisphere
    :param scale:     the scaling factor for the contours
    """

    fig = plt.figure()

    lax = fig.add_subplot(121, aspect='equal')
    lax.set_title("Left eye")
    lax.set_xlabel("Y axis")
    lax.set_ylabel("Z axis")

    for y, z, C11, C12, C21, C22, _, _, _ in lfeats.T:
        C = np.asarray([[C11, C12], [C21, C22]])
        E, V = la.eig(C)
        i = np.argmax(E)

        angle = np.arctan2(V[i,1],V[i,0])

        e = Ellipse(xy=[y, z], width=scale*np.max(E), height=scale*np.min(E),
                                angle=np.rad2deg(angle))
        lax.add_artist(e)
        e.set_clip_box(lax.bbox)
        e.set_facecolor([1, 0, 0])
    limit = Limit * radius
    plt.axis([-limit, limit, -limit, limit])

    rax = fig.add_subplot(122, aspect='equal')
    rax.set_title("Right eye")
    rax.set_xlabel("Y axis")
    for y, z, C11, C12, C21, C22, _, _, _ in rfeats.T:
        C = np.asarray([[C11, C12], [C21, C22]])
        E, V = la.eig(C)
        i = np.argmax(E)

        angle = np.arctan2(V[i,1],V[i,0])

        e = Ellipse(xy=[y, z], width=scale*np.max(E), height=scale*np.min(E),
                                angle=np.rad2deg(angle))
        rax.add_artist(e)
        e.set_clip_box(rax.bbox)
        e.set_facecolor([0, 1, 0])

    limit = Limit * radius
    plt.axis([-limit, limit, -limit, limit])

    plt.show()


# 3D plots

def points3d(points, title='', zdir='z', s=20, c='y', marker='.', radius=1.):
    """
    Plots the given ommatidia as points in the 3D space.

    :param points:    the 3D points of the ommatidia
    :param title:     the title
    :param zdir:      which direction to use as z (x, y or z) when
                      plotting a 2D set.
    :param s:         size in points^2. It is a scalar or an array of the same length
                      as x and y.
    :param c:         colour
    :param marker:    the type of marker (matplotlib format)
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x, y, z in points.T:
        ax.scatter(x, y, z, zdir=zdir, s=s, c=c, marker=marker)

    ax.set_title(title)
    limit = Limit * radius
    ax.set_xlim3d(-limit, limit)
    ax.set_ylim3d(-limit, limit)
    ax.set_zlim3d(-limit, limit)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()
