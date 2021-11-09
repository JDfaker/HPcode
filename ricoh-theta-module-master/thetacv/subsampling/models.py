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
# Models of different subsampling approaches.
#

import numpy as np
import numpy.linalg as la
import csv, os, yaml
from warnings import warn

import thetacv.subsampling.splot as plt

# get path of the script
cpath = os.path.dirname(os.path.abspath(__file__)) + '/'


# load parameters
with open(cpath + '/params.yaml', 'rb') as f:
    params = yaml.safe_load(f)
    sparams = params['subsampling']


class Model(object):
    """
    Interface class for the subsampling models.
    """

    # biological radius of the crab's eye
    EyeRadius = 1.

    def __init__(self, radius=EyeRadius, rebuild=False):
        """
        Initialises the model with using the radius of the fisheye.

        :param radius:      the radius of the fisheye
        :param rebuild:     rebuilds the model instead of loading it
        """
        self.radius = radius


    @staticmethod
    def save(features, filename):
        """
        Saves the features of the model in a CSV file.
        """
        with open( filename, 'wb' ) as f:
            writer = csv.writer(f, delimiter=',')
            for row in features:
                writer.writerow(row)


    @staticmethod
    def load(filename):
        """
        Loads the features of the model from a CSV file.
        """
        rows = []
        with open( filename, 'r' ) as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                 rows.append(np.asarray(row).astype(np.float32))
        return np.asarray(rows)


class SmolkaModel(Model):
    """
    The model provided by Jochen Smolka and Jan Hemmi.
    Retrieved by their publicated paper 'Topography of vision and behaviour'.
    """
    # name: this is used to name the related files
    Name = 'smolka'
    # this file contains data for the right eye
    DefaultFilename = cpath + sparams['filepattern'] % Name

    def __init__(self, radius=None, rebuild=False):
        """
        :param radius:      the radius of the eye
        :param rebuild:     rebuilds the model instead of loading it.
                            this does not affect this class
        """
        super(SmolkaModel, self).__init__(radius)
        phi, theta, rho = Model.load(SmolkaModel.DefaultFilename).T
        phi = aziadj(np.deg2rad(phi + 90))
        theta = eleadj(np.deg2rad(theta))

        if radius:
            rho = radius * np.ones(theta.shape)
        else:
            rho /= rho.max()

        self.rfeatures = np.vstack([theta, phi, rho]).T
        self.lfeatures = self.rfeatures.copy()
        self.lfeatures[:,1] *= -1


    def get_features(self):
        """
        Returns the features of both eyes in a vector, i.e. all the 3D points
        of the ommatidia in a spherical coordinate system.
        """
        r = self.rfeatures[:,1] <= 0
        l = self.lfeatures[:,1] >= 0
        return np.append(self.lfeatures[l], self.rfeatures[r], axis=0)




class CustomModel(Model):
    """
    A custon model of the ommatidia features, which is drawn from a
    combination of the John Layne and Jochen Smolka models.
    """
    # name: this is used to name the related files
    Name = 'custom'

    DefaultFilename = cpath + sparams['filepattern'] % Name

    HorizonIndex    = sparams['row']['horizon']['index'] # frontal
    VFrontalSamples = sparams['col']['length'] # vertical number of samples (frontal)
    HHorizonSamples = sparams['row']['length'] # half horizon number of samples (lateral)

    # Mean vertical interommatidial angle
    # Frontal =  1.634557
    # Rear =  2.17940933
    # Total =  1.90698317
    HFront = sparams['row']['horizon']['front-io-angle']
    HSide  = sparams['row']['horizon']['side-io-angle']
    HRear  = sparams['row']['horizon']['front-io-angle']

    DFront = sparams['row']['dorsal']['front-io-angle']
    DSide  = sparams['row']['dorsal']['side-io-angle']


    def __init__(self, radius=Model.EyeRadius, rebuild=False):
        """
        Initialises the model with using the radius of the fisheye.

        :param radius:      the radius of the fisheye
        :param rebuild:     rebuilds the model instead of loading it.
        """
        super(CustomModel, self).__init__(radius)

        if os.path.exists(CustomModel.DefaultFilename) and not rebuild:
            # load the model
            self.features = Model.load(CustomModel.DefaultFilename)
        else:
            # build and save the model
            (l_hemi, l_hil), (r_hemi, r_hil) = self.construct_3Dpt_sphere()

            lf, rf = [], []
            for (ly, lx, lz), (ry, rx, rz) in zip(l_hemi.T, r_hemi.T):
                ltheta, lphi, _ = vec2sph(lx, ly, lz)
                rtheta, rphi, _ = vec2sph(rx, ry, rz)

                lf.append(np.asarray([ltheta, lphi, self.radius]))
                rf.append(np.asarray([rtheta, rphi, self.radius]))

            self.features = np.append(lf, rf, axis=0)

            Model.save(self.features, CustomModel.DefaultFilename)


    def get_features(self):
        """
        Returns the features of both eyes in a vector, i.e. all the 3D points
        of the ommatidia in a spherical coordinate system.
        """
        return self.features


    def get_ommatidia_gaussians(self, cscale=1., show=False):
        """
        Returns the gaussian features of the two eyes separately.
        Features: mu_x, mu_y, C_11, C_12, C_21, C_22, theta, phi, s

        where:

        - mu = [mu_x, mu_y] is the centre of the gaussian
        - C = [[C_11, C_12], [C_21, C_22]] is the covariance matrix
        - theta is the elevation of the ommatidium
        - phi is the azimuth of the ommatidium
        - s is the horizontal interommatidial angle in the two axes

        :param cscale:    is the scale of the covariance matrix of the Gaussians
        """

        # we set the covariance matrix of the Gaussians to be the same
        # for all the ommadia in the 3D space (a unit disk)
        Sigma = cscale * np.asarray([[1,0,0], [0,1,0], [0,0,0]])
        (l_hemi, l_hil), (r_hemi, r_hil) = \
                self.construct_3Dpt_sphere(show=False)

        l_gs, r_gs = [], []     # Gaussian parameters
        l_sc, r_sc = [], []     # Spherical coordinates
        for (ly, lx, lz), (ry, rx, rz) in zip(l_hemi.T, r_hemi.T):
            ltheta, lphi, _ = vec2sph(lx, ly, lz)
            rtheta, rphi, _ = vec2sph(rx, ry, rz)
            lphi = -lphi # because we assume the right hemisphere

            lR = get_rotation_matrix(ltheta, lphi)
            rR = get_rotation_matrix(rtheta, rphi)

            lC = lR.dot(Sigma).dot(lR.T)
            rC = rR.dot(Sigma).dot(rR.T)

            lx, ly, lz = sph2vec(ltheta, lphi, self.radius)
            rx, ry, rz = sph2vec(rtheta, rphi, self.radius)
            l_gs.append(np.append([ly, lz], lC[:2,:2]))
            r_gs.append(np.append([ry, rz], rC[:2,:2]))

            l_sc.append(np.asarray([ltheta, lphi]))
            r_sc.append(np.asarray([rtheta, rphi]))

        l_features = np.vstack([np.vstack(l_gs).T, np.vstack(l_sc).T, l_hil])
        r_features = np.vstack([np.vstack(r_gs).T, np.vstack(r_sc).T, r_hil])

        if show:
            plt.twin_hemishpere_gauss(l_features, r_features, radius=self.radius)

        return l_features, r_features


    def get_side_slice(self,
            fscale=sparams['col']['fscale'],
            rscale=sparams['col']['rscale'],
            rsang=sparams['col']['rsang'],
            rrang=sparams['col']['rrang'],
            show=False):
        """
        Returns the line coefficients that connect the frontal and rear
        ommatidia, the radius and the respective nominal elevation for each
        ommatidium on this slice.

        :param radius:    the radius of the sphere (eye)
        :param fscale:    frontal scaling
        :param rscale:    rear scaling (wrt frontal)
        :param rsang:     rear starting angle
        :param rrang:     rear angular range
        """

        fangle = np.zeros((CustomModel.VFrontalSamples, 1))     # frontal cumulative angle
        rangle = np.zeros((CustomModel.VFrontalSamples, 1))     # rear cumulative angle
        for n in xrange(CustomModel.VFrontalSamples):
            if n > 0:
                fprev = fangle[n-1]
                rprev = rangle[n-1]
            else:
                fprev = 0
                rprev = 0
            fangle[n] = fprev + \
                get_vertical_interommatidial_angle(n, s=fscale)
            rangle[n] = rprev + \
                get_vertical_interommatidial_angle(n, s=fscale * rscale)

        # elevation of every horizontal line (horizon is zero)
        fangle = np.deg2rad(fangle - fangle[CustomModel.HorizonIndex])
        rangle = np.deg2rad(rangle - rangle[CustomModel.HorizonIndex])

        sangle = np.deg2rad(rsang)            # starting angle
        eangle = np.deg2rad(rsang-rrang)     # ending angle

        rangle = rangle[
            np.all(np.asarray((rangle >= eangle, rangle <= sangle)), axis=0)
        ]

        # compute coordinates
        f_eye_x = np.cos(fangle) * self.radius
        f_eye_z = np.sin(fangle) * self.radius
        r_eye_x = -np.cos(rangle) * self.radius
        r_eye_z = np.sin(rangle) * self.radius

        p_eye_x = np.append(f_eye_x, r_eye_x[::-1])
        p_eye_z = np.append(f_eye_z, r_eye_z[::-1])

        # get only the points which have positive z value (dorsal)
        d_eye_x = p_eye_x[p_eye_z>=0]
        d_eye_z = p_eye_z[p_eye_z>=0]

        # nominal elevation
        n_elevation_line = fangle[fangle>=0]

        # create  and plot the correspondences
        lines_coef_radius = []
        for i, j in zip(range(0, len(d_eye_x)/2),
                        range(len(d_eye_z)-1, len(d_eye_z)/2, -1)):
            # create the corresponding pairs
            # find the equation for the line between each pair of corresponding points
            a, b = np.polyfit([d_eye_x[i],d_eye_x[j]],[d_eye_z[i],d_eye_z[j]], 1)
            # save the coefficients of the line y = ax + b, the radius of the respective horizontal slice
            # and the nominal elevation of the slice and the center of the line
            di = np.asarray([d_eye_x[i], d_eye_z[i]])
            dj = np.asarray([d_eye_x[j], d_eye_z[j]])
            d = la.norm(di - dj)
            dc = (di + dj)/2
            lines_coef_radius.append([a, b, d/2, dc, n_elevation_line[i]])

        # mirror the dorsal to crate the ventral hemisphere
        for i in xrange(sum(f_eye_z < 0)):
            a, b, r, dc, e = lines_coef_radius[i]
            # if dc[1] == -dc[1]:
            #     continue
            dc[1] = -dc[1]
            lines_coef_radius.append([-a, -b, r, dc, -e])

        if show:
            plt.side_slice(p_eye_x, p_eye_z, radius=self.radius)

        # the coefficients of the line below which there are no ommatidia
        # defined from the lower point of the front and rear parts respectively
        a, b = np.polyfit([p_eye_x[0],p_eye_x[-1]],[p_eye_z[0],p_eye_z[-1]], 1)

        return lines_coef_radius, (a, b)


    def get_horizontal_slice(self, radius,
                hsamples = HHorizonSamples,
                h_front=HFront, h_side=HSide, h_rear=HRear):
        """
        Returns the line coefficients that connect the frontal and rear
        ommatidia, the radius and the respective nominal elevation for each
        ommatidium on this slice.

        :param radius:    the radius of the sphere (eye)
        :param h_front:   interommatidial angle scaling (wrt horizon)
        :param h_side:    rear starting angle
        :param h_rear:    rear angular range
        """

        col = np.linspace(0, hsamples, hsamples)
        hangle = get_horizontal_interommatidial_angle(col,
                        h_front=h_front, h_side=h_side, h_rear=h_rear)
        hangle = np.deg2rad(hangle)

        cangle = np.zeros(hangle.shape)
        for n in xrange(len(cangle)):
            if n > 0:
                prev = cangle[n-1]
            else:
                prev = 0
            cangle[n] = prev + hangle[n]

        # azimuth (centring in lateral angle)
        cangle = cangle - cangle[len(col) / 2]

        return cangle, hangle


    def get_all_horizontal_slices(self, lines_info,
                d_front=DFront, d_side=DSide, show=False):
        """
        Returns a list of lists with all the slices and the corresponding
        ommatidia position on the sphere

        :param lines_info:    line information generated from get_side_slice funct
        :param d_front:   front interommatidial angle for dorsal area  wrt to sphere center
        :param d_side:    side interommatidial angle for dorsal area  wrt to sphere center
        """

        # interpolate between the theoritical interomatidia angle on the horizon
        # and at the dorsal of the crab's eye
        col = np.linspace(0, len(lines_info) -1, len(lines_info))
        # front and rear area of the eye
        h_fronts = np.interp(col * 90. / len(col), [0, 90], [CustomModel.HFront, d_front])
        # latteral area of the eye
        h_sides = np.interp(col * 90. / len(col), [0, 90], [CustomModel.HSide, d_side])

        # 'i' is for interommatidial
        all_hor_slices, all_hori_slices = [], []
        odd = False
        for (a, b, slice_r, cl, n_elevation), (h_front, h_side) in zip(lines_info, zip(h_fronts, h_sides)):
            odd = not odd

            # based on the theoritical interomatidia angle, compute the interomatidia
            # angle for each horizontal slice, throught the length of the arc

            # front and rear area of the eye
            f_arclen = angle_rad_2_arc_len(h_front, self.radius)
            f_hor_angle = arc_rad_2_angle(f_arclen,slice_r)

            # side area of the eye
            l_arclen = angle_rad_2_arc_len(h_side, self.radius)
            l_hor_angle = arc_rad_2_angle(l_arclen,slice_r)

            # compute the mean interommatidia angle of the horizontal slice
            # mhor_angle = (2*f_hor_angle + l_hor_angle)/3.
            # find the number of samples required for the slice (size, inter-angle)
            n_samples, mhor_angle = \
                    num_sample_hslice(l_hor_angle, f_hor_angle)

            # compute slice
            hangles, hiangles = self.get_horizontal_slice(radius=slice_r,
                    # we add 2 samples in the odd slides to fill the gap
                    # occured by the shifting
                    hsamples = n_samples + 2*int(odd),
                    h_front=f_hor_angle, h_side=l_hor_angle, h_rear=f_hor_angle)

            # shift the position of the ommatidia to create blister like shape
            if odd:
                theta = np.deg2rad(mhor_angle)/2.
                hangles = hangles - theta
                # erase the samples that shifted too much
                keep = hangles >= -np.pi/2 - np.deg2rad(1)
                hangles = hangles[keep]
                hiangles = hiangles[keep]

            # generate positions oof the ommatidia
            h_eye_x = np.cos(hangles) * slice_r
            h_eye_y = np.sin(hangles) * slice_r
            hi_eye_x = np.cos(hiangles) * slice_r
            hi_eye_y = np.sin(hiangles) * slice_r

            all_hor_slices.append([h_eye_x, h_eye_y])
            all_hori_slices.append(hiangles)

        if show:
            plt.top_view(all_hor_slices, radius=self.radius)

        return (all_hor_slices, all_hori_slices)


    def get_hemisphere(self, lines_info, hor_lines_info,
                        a_cut=None, b_cut=None, show=False):
        """
        Returns a list with the 3D coordinates of all the ommatidia in
        a hemisphere [y,x,z] axis, as well as their interomatidial angles.

        :param lines_info:          line information generated from
                                    get_side_slice funct
        :param hor_lines_info:      half-circles information generated from
                                    get_all_horizontal_slices funct
        """

        omma_3D_pos, hil = np.asarray([]), np.asarray([])

        for (h_eye_x,h_eye_y),hi_angle, \
                (a, b, slice_r, cl, n_elevation) in \
                zip(hor_lines_info[0], hor_lines_info[1], lines_info):
            polynomial = np.poly1d([a,b])
            # swift the ommatidia towards the center of the respective slice
            h_eye_y  = (h_eye_y + cl[0])
            h_eye_z = polynomial(h_eye_y)

            # when we set a cut plane, we drop all the ommatidia below this plane
            if a_cut and b_cut:
                f = a_cut * h_eye_y + b_cut - h_eye_z < 0
            else:
                f = np.ones(h_eye_y.shape) > 0

            new_o3p = np.vstack([h_eye_y[f], h_eye_x[f], h_eye_z[f]])
            new_hil = hi_angle[f]
            if omma_3D_pos.size == 0:
                omma_3D_pos = new_o3p
                hil = new_hil
            else:
                omma_3D_pos = np.append(omma_3D_pos, new_o3p, axis=1)
                hil = np.append(hil, new_hil)

        if show:
            plt.points3d(omma_3D_pos, title='Left Eye', radius=self.radius)

        # this is a hack due to an empty line at the end
        return omma_3D_pos, hil


    def get_two_hemispheres(self, lines_info, hor_lines_info,
                    a_cut=None, b_cut=None, show=False):
        """
        Returns two lists with the 3D coordinates of all the ommatidia in
        a hemisphere [y,x,z] axis, as well as their interomatidial angles.

        :param lines_info:          line information generated from
                                    get_side_slice funct
        :param hor_lines_info:      half-circles information generated from
                                    get_all_horizontal_slices funct
        """

        l_hemi, hil = self.get_hemisphere(lines_info, hor_lines_info,
                    a_cut=a_cut, b_cut=b_cut, show=False)
        r_hemi = l_hemi.copy()
        l_hemi[1] *= -1

        if show:
            sphere = np.append(l_hemi, r_hemi, axis=1)
            plt.points3d(sphere, title='Combound Eyes', radius=self.radius)

        # r_hemi[0] *= -1

        return (l_hemi, hil), (r_hemi, hil)


    def construct_3Dpt_sphere(self, show=False):
        """
        Returns a list of lists with the 3D coordinates of all the ommatidia in
        a hemiphere [y,x,z] axis

        Constructs the hemisphere utilising the functions of the class
        """
        lcr, (a_cut, b_cut) = self.get_side_slice(show=show)
        hinfo = self.get_all_horizontal_slices(lcr,show=show)
        l_hemi, r_hemi = self.get_two_hemispheres(lcr, hinfo, a_cut=a_cut, b_cut=b_cut,
                show=show)
        return l_hemi, r_hemi


# in the horizon is num_sample = 75.
# c is the number of interpolated steps(points)
def get_horizontal_interommatidial_angle(c, h_front=CustomModel.HFront,
                    h_side=CustomModel.HSide, h_rear=CustomModel.HRear):
    """
    Computes the horizontal interommatidial angle from the given the column
    indices.

    :param c:         the column index
    :param h_front:   the frontal interommatidial angle
    :param h_side:    the lateral (side) interommatidial angle
    :param h_rear:    the rear interommatidial angle
    """
    fhp = [h_front, h_side, h_rear]
    return np.interp(c * 180 / c.size, [0, 90, 180], fhp)


def get_vertical_interommatidial_angle(r, s = 1.):
    """
    Computes the vertical interommatidial angle from the given rows and
    scale.

    :param r:     the row
    :param s:     the scale
    """
    rr = r - CustomModel.HorizonIndex
    return (0.001334 * rr * rr + 0.52) / s


def angle_rad_2_arc_len(angle, radius):
    """
    Compute the legthn of an arc from angle and radius

    :param angle:   the angle of the arc
    :param radius:  the radius of the circle
    """
    radangle = np.deg2rad(angle) # convert to radian
    #radangle = (angle) # convert to radian
    return radangle*radius       # return the arc length


def arc_rad_2_angle(arclen, radius):
    """
    Computes the angle from arc and radius.

    :param arclen:  the length of the arc
    :param radius:  the radius of the circle
    """
    # return the angle inf degrees
    return np.rad2deg(arclen/radius)


def num_sample_hslice(min_ang, max_ang):
    """
    Computes the number of samples per horizontal slice based in the
    mean interomatidia angle.

    :param min_ang:   the minimum (in the lateral area) interommatidial angle
    :param max_ang:   the maximum (in the frontal and rear area) interommatidial angle
    """

    # [front, lateral, rear]
    sx = [0, 90, 180]                   # source x
    sy = [max_ang, min_ang, max_ang]    # source y
    # 200 equally ranged samples in [0, 180]
    x = np.linspace(0, 180, 200)        # sample indices
    y = np.interp(x, sx, sy)            # sample interommatidial angles

    m_y = np.mean(y)
    return round(180/m_y), m_y        # approximation of number of samples


def get_rotation_matrix(x, y, z=None):
    """
    Return a 3D rotation matrix using the input coordinates

    :param x: if z is None x is the elevation
    :param y: if z is None y is the azimuth
    :param z: None for spherical coordinates
    """
    if z:
        # convert to spherical coordinates
        theta, phi = vec2sph(x, y, z)
    else:
        theta, phi = x, y

    # -pi / 2 is to recenter in 0 (from pi / 2)
    # abs is to mirror the points for the two hemispheres
    #   the left hemisphere is not affected because it contains only positive
    #   numbers, while the right hemisphere is mirrored as it has only negative
    phi = np.pi / 2 - np.abs(phi)

    # convert to rotation matricies
    x, y, z = theta, phi, 0

    Rx = np.asarray([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
            ])
    Ry = np.asarray([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
            ])
    Rz = np.asarray([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
    ])

    # return combined rotation matrix
    # The order of the multiplication is Rx, Ry, Rz
    # The rotations are applied in relative manner,
    # i.e. each rotation changes the coordinate system
    return Rz.dot(Ry).dot(Rx)


def get_shperical_coords(x, y, z):
    """
    Return the spherical coordinates of a 3D point.

    :param theta: elevation [-pi/2, pi/2]
    :param phi:   azimuth [-pi, 0]
    """
    warn("This method is deprecated. Use vec2sph instead.")

    v = np.asarray([x, y, z])
    rho = la.norm(v)
    x, y, z = v / rho

    phi = np.arccos(z)         # 0 is frontal line
    theta = np.arctan2(y, x)   # 0 is horizon

    # conditions to restrict the angles to correct quadrants
    if theta > np.pi / 2:
        theta = np.pi - theta
    elif theta < -np.pi / 2:
        theta = -np.pi - theta

    # -pi is for the right hemisphere [-pi, 0]
    return theta, phi - np.pi, rho


def vec2sph(x, y, z):
    """
    Return the spherical coordinates of a 3D point.

    :param theta: elevation [-pi/2, pi/2]
    :param phi:   azimuth [-pi, 0]
    """
    vec = np.asarray([x, y, z])
    rho = la.norm(vec)   # length of the radius

    if rho == 0:
        rho = 1.
    v = vec / rho             # normalised vector

    phi = np.arctan2(v[0], v[1])        # azimuth
    theta = np.pi / 2 - np.arccos(v[2])   # elevation

    # theta, phi = sphadj(theta, phi)     # bound the spherical coordinates
    return np.asarray([theta, phi, rho])



def sph2vec(theta, phi, rho=1.):
    """
    Transforms the spherical coordinates to a cartesian 3D vector.

    :param theta: elevation
    :param phi:   azimuth
    :param rho:   radius length
    """

    x = rho * (np.sin(phi) * np.cos(theta))
    y = rho * (np.cos(phi) * np.cos(theta))
    z = rho * np.sin(theta)

    return np.asarray([x, y, z])


# conditions to restrict the angles to correct quadrants
def eleadj(theta):
    """
    Adjusts the elevation in [-pi, pi]

    :param theta:   the elevation
    """
    theta, _ = sphadj(theta=theta)
    return theta


def aziadj(phi):
    """
    Adjusts the azimuth in [-pi, pi].

    :param phi: the azimuth
    """
    _, phi = sphadj(phi=phi)
    return phi


def sphadj(theta=None, phi=None,
           theta_min=-np.pi/2, theta_max=np.pi/2,   # constrains
           phi_min=-np.pi, phi_max=np.pi):
    """
    Adjusts the spherical coordinates using the given bounds.

    :param theta:     the elevation
    :param phi:       the azimuth
    :param theta_min: the elevation lower bound (default -pi/2)
    :param theta_max: the elevation upper bound (default pi/2)
    :param phi_min:   the azimuth lower bound (default -pi)
    :param phi_max:   the azimuth upper bound (default pi)
    """

    # change = np.any([theta < -np.pi / 2, theta > np.pi / 2], axis=0)
    if np.all(theta):
        if (theta >= theta_max).all():
            theta = np.pi - theta
            if np.all(phi):
                phi += np.pi
        elif (theta < theta_min).all():
            theta = -np.pi - theta
            if np.all(phi):
                phi += np.pi
        elif (theta >= theta_max).any():
            theta[theta >= theta_max] = np.pi - theta[theta >= theta_max]
            if np.all(phi):
                phi[theta >= theta_max] += np.pi
        elif (theta < theta_min).any():
            theta[theta < theta_min] = -np.pi - theta[theta < theta_min]
            if np.all(phi):
                phi[theta < theta_min] += np.pi

    if np.all(phi):
        while (phi < phi_min).all():
            phi += 2 * np.pi
        while (phi >= phi_max).all():
            phi -= 2 * np.pi
        while (phi < phi_min).any():
            phi[phi < phi_min] += 2 * np.pi
        while (phi >= phi_max).any():
            phi[phi >= phi_max] -= 2 * np.pi


    return theta, phi
