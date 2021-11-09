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
# These classes are used for preprocessing of the images aiming to:
# - split the images in left and right eyes
# - extract features
# - unwarp panoramic and 360 degrees imgages
# - tranform and distort the visual field, i.e. wrt crab's field of view
#

import csv, os, cv2, time, yaml
import pickle
import numpy as np
from scipy.stats import multivariate_normal as mvgauss

from thetacv.subsampling.models import SmolkaModel, sph2vec
from thetacv.capture.utils import mix_kernels, RotBlur, blend_mix_imgs

# get path of the script
filepath = os.path.dirname(os.path.abspath(__file__)) + '/'


# load parameters
with open(filepath + 'params.yaml', 'rb') as f:
    params = yaml.safe_load(f)
    cparams = params['capture']


# Default width and height of every lense's image
Width, Height = cparams['frame']['width'], cparams['frame']['height']
Channels = cparams['frame']['channels']
Scale = cparams['frame']['scale']
RadiusScale = cparams['radius']['scale']


class NonePreprocess(object):
    """
    Default preprocess method, which just crops the useful image from the one
    that is obtained, and separetes the two fisheye images from the combined
    one.
    """

    def __init__(self, h_offset=5, v_offset=0, bw=False, colour_spec=True,
                 img_size=(int(Width*Scale), int(Height*Scale), int(Channels))):
        """
        Initialises the default offset of the image.

        :param h_offset:    the horizontal offset of the image
        :param v_offset:    the vertical offset of the image
        :param img_size:    the size of the image for each eye
        """
        self.v_offset = v_offset
        self.h_offset = h_offset
        self.width, self.height, self.channels = img_size
        self.ckernel = np.ones((5,5),np.uint8)
        self.bw = bw
        self.colour_spec = colour_spec


        with open(filepath + '/params.yaml', 'rb') as f:
            params = yaml.safe_load(f)
            self.skyblue = np.asarray(params['simulation']['colour']\
                                                          ['skyblue'])
            self.sandybrown = np.asarray(params['simulation']['colour']\
                                                             ['sandybrown'])
            self.dummyblack = np.asarray(params['simulation']['colour']\
                                                             ['dummyblack'])
            self.burrowbrown = np.asarray(params['simulation']['colour']\
                                                              ['burrowbrown'])

        if self.channels == 1:
            self.skyblue = self.skyblue.mean()
            self.sandybrown = self.sandybrown.mean()
            self.dummyblack = self.dummyblack.mean()
            self.burrowbrown = self.burrowbrown.mean()


    def preprocess(self, data):
        """
        Crops the useful images and separates the combined fisheye images.

        :param data:    the unprocessed (raw) image, which contains information
                        of both eyes.
        """

        data = data[
            self.h_offset:self.h_offset+Height,
            self.v_offset:self.v_offset+2*Width
        ]
        img_size=(2*self.width, self.height)
        org_size = np.asarray(data.shape[:2][::-1], dtype=np.float32)
        fx, fy = np.asarray(img_size).astype(float) / org_size.astype(float)
        data = cv2.resize(data, (0, 0), fx=fx, fy=fy)

        eyes = np.split(data.copy(), 2, axis=1)
        leye = np.rot90(eyes[0], 3)
        reye = np.rot90(eyes[1], 1)
        #####
        if self.colour_spec:
            data_ = np.concatenate((leye, reye), axis = 1)
            data_ = self.detect_filter(data_) * 255.
            leye, reye = np.split(data_, 2, axis=1)

        return (data, leye, reye)


    def detect_filter(self, im):
        """
        Applies a colour filter to genrate the view of the world as the crab
        perceives it. Uniform light colour background and the predator is
        really distict.

        :param data:    the unprocessed (raw) image, which contains information
                        of both eyes.
        """

        # thresholds for the colour filter in HSV scale
        # detecting violet colour
        lower_violet = np.array([120, 80, 100], np.uint8)
        upper_violet = np.array([170, 255, 255], np.uint8)

        # find the colors within the specified boundaries and apply
        # the mask
        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        vmask = cv2.inRange(hsv_image, lower_violet, upper_violet)

        """
        Generate the image of the view of the crab : White background and
        predator as a contrastive black object
        """

        # apply morphological operation to unite the colourful detected pixels
        closing = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, self.ckernel)
        bw = cv2.bitwise_not(closing)

        # add axis so the concatenatations done later can work
        bw = bw[...,np.newaxis]

        # return the black and white view of the world
        if self.bw: return np.concatenate([bw] * self.channels, axis=2)

        # get the size of the image
        u, v, _ = bw.shape

        # pixel that defines the horizon
        horizon_index = u / 2

        # colour the appropriate pixels to generate the simulation colours
        up = bw[:horizon_index,:] / 255.
        lw = bw[horizon_index:,:] / 255.

        # colour the upper hemishpere view
        skydum = self.skyblue - self.dummyblack
        # the colours are applied in every channel (normally BGR)
        cup = np.concatenate([up * skydum[c] + self.dummyblack[c] \
                        for c in range(self.channels)], axis=2)

        # colour the lower hemishpere view
        sandburr = self.sandybrown - self.burrowbrown
        # the colours are applied in every channel (normally BGR)
        clw = np.concatenate([lw * sandburr[c] + self.burrowbrown[c] \
                        for c in range(self.channels)], axis=2)

        # combine above and below horizon view
        cim = np.concatenate((cup, clw), axis = 0)

        return cim


    def SaveMap(self, filename):
        """
        Rebuilds and saves the current maps (x and y) in a single file.
        It uses the default (internal) path for the maps-file.

        :param filename:    the file path where you want to save the maps.
        """
        self.xmap, self.ymap = self.BuildMap()
        pickle.dump( (self.xmap, self.ymap), open( filename, "wb" ) )



class Dewarp(NonePreprocess):
    """
    Dewarp prepcrocessing class. Tranforms every fisheye (circular) image to
    a square image, with as less as possible distortion.
    """
    DefaultMapsFilename = filepath + cparams['warp']['filename']


    def __init__(self, bw=False, colour_spec=True, rebuild_map=False):
        """
        :param rebuild_map:     whether to rebuild the map during the
                                initialisation or not
        """
        super(Dewarp, self).__init__(bw=bw, colour_spec=colour_spec)

        # check if the file exists
        if (os.path.exists(Dewarp.DefaultMapsFilename)):
            self.xmap, self.ymap = \
                    pickle.load( open( Dewarp.DefaultMapsFilename, "rb" ) )

        if rebuild_map or not os.path.exists(Dewarp.DefaultMapsFilename):
            self.SaveMap(Dewarp.DefaultMapsFilename)


    def preprocess(self, data):
        """
        Tranforms every fisheye (circular) image to a square image, with as
        less as possible distortion.

        :param data:    the raw data (both eyes)
        """
        data, leye, reye = super(Dewarp, self).preprocess(data)

        fi, ti = int(0.1*self.width), int(0.9*self.height)

        leye = cv2.remap(leye, self.xmap, self.ymap, cv2.INTER_LINEAR)
        leye = leye[fi:ti, fi:ti, 0:3]

        reye = cv2.remap(reye, self.xmap, self.ymap, cv2.INTER_LINEAR)
        reye = reye[fi:ti, fi:ti, 0:3]

        return (data, leye, reye)


    def BuildMap(self, Ws=int(Scale*Width), Hs=int(Scale*Height),
                    hfovd=160.0, vfovd=160.0):
        """
        Builds the maps (x and y) using the given parameters.

        :param Ws:      source width
        :param Hs:      source height
        :param hfovd:   the horizontal receptive field of the lense
        :param vfovd:   the vertical receptive field of the lense
        """
        Wd, Hd = Ws, Hs

        # Build the fisheye mapping
        map_x = np.zeros((Hd,Wd),np.float32)
        map_y = np.zeros((Hd,Wd),np.float32)
        vfov = (vfovd/180.0)*np.pi
        hfov = (hfovd/180.0)*np.pi
        vstart = ((180.0-vfovd)/180.00)*np.pi/2.0
        hstart = ((180.0-hfovd)/180.00)*np.pi/2.0
        count = 0
        # need to scale to changed range from our
        # smaller cirlce traced by the fov
        xmax = np.sin(np.pi/2.0)*np.cos(vstart)
        xmin = np.sin(np.pi/2.0)*np.cos(vstart+vfovd)
        xscale = xmax-xmin
        xoff = xscale/2.0
        zmax = np.cos(hstart)
        zmin = np.cos(hfovd+hstart)
        zscale = zmax-zmin
        zoff = zscale/2.0
        # Fill in the map, this is slow but
        # we could probably speed it up
        # since we only calc it once, whatever
        for y in range(0,int(Hd)):
            for x in range(0,int(Wd)):
                count = count + 1
                phi = vstart+(vfovd*((float(x)/float(Wd))))
                theta = hstart+(hfovd*((float(y)/float(Hd))))
                xp = ((np.sin(theta)*np.cos(phi))+xoff)/zscale#
                zp = ((np.cos(theta))+zoff)/zscale#
                xS = Ws-(xp*Ws)
                yS = Hs-(zp*Hs)
                map_x.itemset((y,x),int(xS))
                map_y.itemset((y,x),int(yS))

        return map_x, map_y



class Panorama(NonePreprocess):
    """
    Panorama prepcrocessing class. Tranforms every fisheye (circular) image to
    a panorama image, with as less as possible distortion.
    """
    DefaultMapsFilename = filepath + cparams['pano']['filename']


    def __init__(self, colour_spec=True, bw=False, rebuild_map=False):
        """
        :param rebuild_map:     whether to rebuild the map during the
                                initialisation or not
        """
        super(Panorama, self).__init__(bw=bw, colour_spec=colour_spec)

        if (os.path.exists(Panorama.DefaultMapsFilename)):
            self.xmap, self.ymap = \
                    pickle.load( open( Panorama.DefaultMapsFilename, "rb" ) )

        if rebuild_map or not os.path.exists(Panorama.DefaultMapsFilename):
            self.SaveMap(Panorama.DefaultMapsFilename)


    def preprocess(self, data):
        """
        Tranforms every fisheye (circular) image to a panorama image, with as
        less as possible distortion.

        :param data:    the raw data (both eyes)
        """
        data, leye, reye = super(Panorama, self).preprocess(data)

        leye = np.rot90(leye, 1)
        reye = np.rot90(reye, 3)
        leye = cv2.remap(leye,self.xmap,self.ymap,cv2.INTER_LINEAR)
        reye = cv2.remap(reye,self.xmap,self.ymap,cv2.INTER_LINEAR)
        reye = cv2.flip(reye, 1)
        leye = cv2.flip(np.rot90(leye, 2), 1)

        return (data, leye, reye)


    def BuildMap(self, Ws=int(Scale*Width), Hs=int(Scale*Height)):
        """
        Builds the maps (x and y) using the given parameters.

        :param Ws:  source width
        :param Hs:  source height
        """
        Cx, Cy = int(.5 * Hs), int (.5 * Ws)
        # .46 is the scale of the diameter, to pick the proper part of the
        # fisheye image
        R1, R2 = 1, int (RadiusScale * Ws / 2)

        Wd = 2*(R2-R1)
        Hd = int(2.0*((R2+R1)/2)*np.pi)

        xmap, ymap = np.zeros((Hd, Wd),np.float32), np.zeros((Hd, Wd),np.float32)
        for xD in range(Hd):
            for yD in range(Wd):
                r = (float(yD)/float(Wd))*(R2-R1)+R1
                theta = (float(xD)/float(Hd))*2.0*np.pi
                xS = int(Cx+r*np.sin(theta))
                yS = int(Cy+r*np.cos(theta))
                xmap[xD, yD] = xS
                ymap[xD, yD] = yS

        return xmap, ymap


    @staticmethod
    def blit(limg, rimg, offset, expf=1.2):
        """
        Smoothly blends the right and left images to return a combined image.

        :param limg:    the left image
        :param rimg:    the right image
        :param offset:  how much we want them to overlap
        :param expf:    the exponential factor for the blending
        """
        offset = int(offset)
        final = np.concatenate((limg, rimg[:,offset:]), axis=1)

        loffset = limg.shape[1]-offset
        for i in range(offset):
            factor = np.clip((float(i)**expf)/float(offset), 0.0, 1.0)
            final[:,loffset+i] = \
                (1 - factor) * limg[:, loffset + i] + \
                factor * rimg[:, i]

        return final


class OmmatidiaFeatures(NonePreprocess):
    """
    OmmatidiaFeatures prepcrocessing class.
    Tranforms every fisheye (circular) image to a set of features.
    """

    # Static variables
    DefaultKernelsFilename = filepath + cparams['omma']['kfilename']

    # default covariance scale
    CScale = cparams['omma']['cscale']
    # default covariance of the Gaussian maps
    KScale = cparams['omma']['kscale']
    # default radius of the fisheye (in pixels)
    FRadius = int(RadiusScale * Scale * min(Width, Height) / 2)

    def __init__(self,
                model=SmolkaModel(FRadius), rot_blur=True, colour_spec=True,
                bw=False, fisheye=True, rebuild_kernels=False):
        """
        Initialises the features we want to extract using the given model of
        the ommatidia.

        :param model:     the model of the ommatidia feature's extraction
        :param fisheye:   specifies whether the input is a double fisheye image
        """
        super(OmmatidiaFeatures, self).__init__(bw=bw, colour_spec=colour_spec)
        self.ommodel = model
        self.fisheye = fisheye
        self.apply_rot_blur = rot_blur and fisheye

        kfilename = OmmatidiaFeatures.DefaultKernelsFilename
        if not rebuild_kernels and os.path.exists(kfilename):
            # Gaussian Blur and Rotation Blur kernels
            self.gbkernel, self.rbkernel = \
                            pickle.load( open( kfilename, "rb" ) )
        else:
            self.SaveKernel(kfilename)


        feats = np.split(model.get_features(), 2, axis=0)
        l_feats = feats[0]
        r_feats = feats[1]

        # feats = model.get_features()
        # l_feats = feats[feats[:,1] >= 0]
        # r_feats = feats[feats[:,1] <= 0]

        if fisheye:
            # position on the fish-eye image
            lx, ly, lz = [], [], []
            for theta, phi, rho in l_feats:
                x, y, z = sph2vec(theta, phi, rho)
                lx.append(x)
                ly.append(-y)
                lz.append(z)
            lx, ly, lz = np.asarray(lx), np.asarray(ly), np.asarray(lz)

            rx, ry, rz = [], [], []
            for theta, phi, rho in r_feats:
                x, y, z = sph2vec(theta, phi, rho)
                rx.append(x)
                ry.append(y)
                rz.append(z)
            rx, ry, rz = np.asarray(rx), np.asarray(ry), np.asarray(rz)

            lpx, lpy = self.width/2 - ly, self.height/2 - lz
            rpx, rpy = self.width/2 - ry, self.height/2 - rz
        else:
            # positions on the non-fisheye image
            lpx, lpy = self.sph2pix(-l_feats[:,0], -l_feats[:,1])
            rpx, rpy = self.sph2pix(-r_feats[:,0], -r_feats[:,1])

        self.l_pos = np.vstack(np.int32([lpy, lpx]))
        self.r_pos = np.vstack(np.int32([rpy, rpx]))

        # spherical coordinates
        self.l_sph = self.sph2pix(l_feats[:,0], l_feats[:,1])
        self.r_sph = self.sph2pix(r_feats[:,0], r_feats[:,1])

        # # drop duplicates (using the spherical coordinates)
        # l_sph_c = np.ascontiguousarray(self.l_sph.T).view(
        #     np.dtype((np.void,
        #     self.l_sph.T.dtype.itemsize * self.l_sph.shape[0])))
        # r_sph_c = np.ascontiguousarray(self.r_sph.T).view(
        #     np.dtype((np.void,
        #     self.r_sph.T.dtype.itemsize * self.r_sph.shape[0])))
        #
        # _, l_idx = np.unique(l_sph_c, return_index=True)
        # _, r_idx = np.unique(r_sph_c, return_index=True)
        #
        # # after the unique the sequence of the indexes is changes thus
        # # we resort the indexes to work appropriately
        # l_idx = np.sort(l_idx)
        # r_idx = np.sort(r_idx)
        #
        # self.l_pos = self.l_pos[...,l_idx]
        # self.r_pos = self.r_pos[...,r_idx]
        # self.l_sph = self.l_sph[...,l_idx]
        # self.r_sph = self.r_sph[...,r_idx]

        # Gaussian kernel for bluring the image
        kernel1D = cv2.getGaussianKernel(5, 2)
        self.kernel = kernel1D * kernel1D.T
        self.phi = .5


    def preprocess(self, data, return_img=False, raw=False):
        """
        Returns the features of the data.

        Input:

        :param data:          the input data
        :param return_img:    if true, also return the eyes images as output
        :param raw:           if false, returns the image produced by the
                              get_values function

        Output:

        :return data:          the raw data
        :return lcolour:       the colours of the ommatidia of the left eye
        :return rcolour:       the colours of the ommatidia of the right eye

        Optional:

        :return leye:          the image of the left eye
        :return reye:          the image of the right eye
        """
        if self.fisheye:
            data, leye, reye = super(OmmatidiaFeatures, self).preprocess(data)
        else:
            leye, reye = np.split(data, 2, axis=1)

        leye = self.get_blured(leye)
        reye = self.get_blured(reye)

        lcolour = self.get_colour(leye, self.l_pos)
        rcolour = self.get_colour(reye, self.r_pos)

        if return_img:
            if raw:
                return (data, lcolour, rcolour, leye, reye)
            else:
                return (data, lcolour, rcolour,
                    self.get_values(leye, lcolour, self.l_sph),
                    self.get_values(reye, rcolour, self.r_sph))
        else:
            return (data, lcolour, rcolour)


    def get_blured(self, img):
        """
        Process the image using bluring.

        :param img:     the input image
        """
        # Smooth the image with GaussianBlur
        blur = cv2.filter2D(img, -1, self.kernel)
        if self.channels == 1: blur = blur[...,np.newaxis]
        if not self.fisheye:
            return blur

        if self.apply_rot_blur:
            # Smooth the image with RotationalBlur
            rot = RotBlur(img, self.phi)
        else:
            rot = img

        # Blend the two images with their respective kernels
        img = blend_mix_imgs(blur, self.gbkernel, rot, self.rbkernel)

        return img


    def get_values(self, eye, colour, sph):
        """
        Returns the values of the ommatidia mapped on the fisheye shape.
        All the rest of the image become black.

        :param eye:     the eye image in fisheye format
        :param colour:  the colour values of the ommatidia
        :param sph:     the positions on the image of the respective ommatidia
        """
        # colour = self.get_colour(eye, pos)
        cmap = np.zeros((eye.shape[0], eye.shape[1], 3))

        cmap[sph[0], sph[1], ...] = colour

        return cmap


    def get_colour(self, eye, pos):
        """
        Return the colour of the given position on the given eye in range [0,1]

        :param eye: the eye image (fisheye format)
        :param pos: the pixel position
        """
        return eye[pos[0],pos[1],...] / 255.


    def sph2pix(self, theta, phi):
        """
        Return the related pixel index of the given spherical coordinates

        :param theta:   the elevation
        :param phi:     the azimuth
        """

        x = (self.width - 1) * (phi < 0).astype(int) + np.int32(self.width * phi / np.pi)
        y = np.int32(self.height * (theta + np.pi/2) / np.pi)

        return np.vstack([x, y])


    def pix2sph(self, x, y):
        """
        Return the related spherical coordinates of the given pixel index

        :param theta:   the elevation
        :param phi:     the azimuth
        """

        theta = np.pi * np.float32( y ) / self.height - np.pi / 2

        phi = np.pi * np.float32( x ) / self.width - np.pi

        return np.vstack([theta, phi])


    def BuildKernels(self, Ws=int(Scale*Width), Hs=int(Scale*Height), sigma=KScale*Scale):
        """
        Returns the gaussian and rotational bluring kernels.

        :param Ws:      the width of the kernel
        :param Hs:      the height of the kernel
        :param sigma:   the scaling of the variance for the gaussian kernel
        """
        # Build mixing kernels
        return mix_kernels(Ws, Hs, self.channels, sigma, show=False)


    def SaveKernel(self, filename=DefaultKernelsFilename):
        """
        Rebuilds and saves the current maps (x and y) in a single file.
        It uses the default (internal) path for the maps-file.

        :param filename:    the file path of the kernel
        """
        self.gbkernel, self.rbkernel = self.BuildKernels()
        pickle.dump( (self.gbkernel, self.rbkernel), open( filename, "wb" ) )


    def BuildMap(self, Ws=int(Scale*Width), Hs=int(Scale*Height)):
        """
        Returns a map for the left and the right eye.

        :param Ws:  the width of the map
        :param Hs:  the height of the map
        """
        return np.zeros((Ws, Hs)), np.ones((Ws, Hs))


    def SaveMap(self, filename):
        """
        Rebuilds and saves the current maps (x and y) in a single file.
        It uses the default (internal) path for the maps-file.

        :param filename:    the file path where you want to save the map
        """
        self.lmap, self.rmap = self.BuildMap()
        pickle.dump( (self.lmap, self.rmap), open( filename, "wb" ) )


    def build_featsMap(self):
        """
        Generates a map of the pixels to the ommatidia index according to the
        features map (ommatidia-index to [elevation, azimuth] )

        immap   :   the map which provide the indexing between pixels for
                    an image of 640 x 1280 (W x H) to the ommatidia indexes
        """

        #  get the features map :  Ommatidia ID to elevevatio and azimuth
        feats = self.ommodel.get_features()

        # collect the correct ommatidia indexs for the left and the righ eye
        lfeats = feats[:4735]
        rfeats = feats[4735:]

        # local ommatidia ids to pixels maps for left and right
        lomid2pix  = self.sph2pix(lfeats[:,0],lfeats[:,1]).T
        romid2pix  = self.sph2pix(rfeats[:,0],rfeats[:,1]).T

        # transform the coordinate system of the maps to the upper left corner
        # lomid2pix = 639 - lomid2pix
        # romid2pix = 639 - romid2pix
        # romid2pix[:,0] += 640

        lomid2pix = (self.width - 1) - lomid2pix
        romid2pix = (self.width - 1) - romid2pix
        romid2pix[:,0] += self.width

        # concatenate to on local map from omm id to pixels
        omid2pix = np.append(lomid2pix, romid2pix, axis=0)

        # double width because here we have left and right hemispheres together
        Ws = 2 * self.width
        Hs = self.height

        # build e rectangle to become the map
        rect = (0, 0, Ws, Hs)
        subdiv = cv2.Subdiv2D(rect)

        # fill the map with the info of the local map from omm id to pixels
        tup_omid2pix = tuple(omid2pix)
        subdiv.insert(tup_omid2pix)

        # fill the map in the size of the image according to voronoi
        immap = np.zeros((Hs, Ws), dtype=np.int32)
        (imfacets, imcenters) = subdiv.getVoronoiFacetList([])
        for i, facet in enumerate(imfacets):
            cv2.fillConvexPoly(immap, np.int32(facet), int(i))

        #print "shape of the map " , immap.shape, 'max value', immap.max()

        return immap



class OmmatidiaVoronoi(OmmatidiaFeatures):
    """
    OmmatidiaVoronoi prepcrocessing class.
    Tranforms every fisheye (circular) image to voronoi-like image, produced
    with respect to the the ommatidia features.
    A global blurring method is used before the feature extraction, which is
    applied in the whole image at ones.
    """
    # the file pattern for the voronoi maps.
    # for every subsampling model we build a different map
    MapsFilenamePattern = filepath + cparams['vorn']['filepatt']


    def __init__(self,
                model=SmolkaModel(OmmatidiaFeatures.FRadius), colour_spec=True,
                bw=False, mirror=False, fisheye=True, rebuild_map=False):
        """
        Initialises the featrures of the ommatidia.

        :param model:       the subsampling model for the feature extraction
        :param mirror:      whether the input images are mirrored or not
        :param fisheye:     whether the input images represent fisheye shapes
        :param rebuild_map: if you want to rebuild the map
        """
        super(OmmatidiaVoronoi, self).__init__(model, bw=bw, colour_spec=colour_spec,
                            rebuild_kernels=rebuild_map, fisheye=fisheye)
        self.mirrored = mirror

        # load the maps if they exist
        filename = OmmatidiaVoronoi.MapsFilenamePattern % model.__class__.Name
        if os.path.exists(filename) and not rebuild_map:
            self.lmap, self.rmap = pickle.load( open( filename, "rb" ) )

        # save the maps if they do not exist
        if not os.path.exists(filename) or rebuild_map:
            self.SaveMap(filename)


    def preprocess(self, data):
        """
        Return the raw data and the two eye images dewarped using a voronoi
        diagram and the spherical coordinates of every ommatidium.

        :param data:    the unprocessed (raw) image, which contains information
                        of both eyes.
        """
        data, lcolour, rcolour = \
                super(OmmatidiaVoronoi, self).preprocess(data)

        leye = lcolour[self.lmap,...]
        reye = rcolour[self.rmap,...]

        if self.mirrored:
            return (data, leye[:,::-1,:], reye[:,::-1,:])
        else:
            return (data, leye, reye)


    def get_voronoi(self, eye, points, colour, normalise=True):
        """
        Return a Voronoi-like image of the dewarped fisheye using the spherical
        coordinates of the ommatidia.

        :param eye:       the uprocessed image of the eye
        :param points:    the spherical coordinates of the ommatidia
        :param colour:    the colour value of the ommatidia
        :param normalize: flag to normalise or not the values of the returned image
        """
        rect = (0, 0, eye.shape[1], eye.shape[0])
        subdiv = cv2.Subdiv2D(rect);

        pts = tuple(np.asarray([points[1],points[0]]).T)
        subdiv.insert(pts)

        # Allocate space for Voronoi Diagram
        voronoi = np.zeros(eye.shape, dtype = eye.dtype)
        self._draw_voronoi(voronoi,subdiv,colour)

        return voronoi / ( 255. if normalise else 1.)


    # Draw voronoi diagram
    def _draw_voronoi(self, img, subdiv, colours) :
        """
        Draws the voronoi-like image on the given image.

        :param img:     the image that will be used as canvas
        :param subdiv:  the subdivision object which describes the facets of the
                        diagram
        :param colours: the colours of the facets
        """
        ( facets, centers) = subdiv.getVoronoiFacetList([])
        # print len(facets), colour.shape
        for facet, colour in zip(facets, colours):
            ifacet = np.int32(facet)
            colour = (int(colour[0] * 255),int(colour[1] * 255),int(colour[2] * 255))

            cv2.fillConvexPoly(img, ifacet, colour);
            ifacets = np.array([ifacet])
            cv2.polylines(img, ifacets, True, (0, 0, 0), 1)
            # cv2.circle(img, (centers[i][0], centers[i][1]), 1, (0, 0, 0), -1)


    def BuildMap(self, Ws=int(Scale*Width), Hs=int(Scale*Height)):
        """
        Builds the voronoi map for both eyes. In every position on the map we
        set the index of the ommatidium the correspond to that region.

        :param Ws:  the width of the map
        :param Hs:  the height of the map
        """

        rect = (0, 0, Hs, Ws)
        l_subdiv = cv2.Subdiv2D(rect)
        r_subdiv = cv2.Subdiv2D(rect)

        # l_pts = tuple(np.asarray([Ws-self.l_sph[0]-1,Hs-self.l_sph[1]-1]).T)
        # r_pts = tuple(np.asarray([Ws-self.r_sph[0]-1,Hs-self.r_sph[1]-1]).T)
        l_pts = tuple(np.asarray([Ws-self.l_sph[0]-1,Hs-self.l_sph[1]-1]).T)
        r_pts = tuple(np.asarray([Ws-self.r_sph[0]-1,Hs-self.r_sph[1]-1]).T)
        l_subdiv.insert(l_pts)
        r_subdiv.insert(r_pts)

        l_map = np.zeros((Ws, Hs), dtype=np.int32)
        r_map = np.zeros((Ws, Hs), dtype=np.int32)
        (l_facets, l_centers) = l_subdiv.getVoronoiFacetList([])
        (r_facets, r_centers) = r_subdiv.getVoronoiFacetList([])

        for facet, i in zip(l_facets, range(len(l_centers))):
            cv2.fillConvexPoly(l_map, np.int32(facet), int(i))
        for facet, i in zip(r_facets, range(len(r_centers))):
            cv2.fillConvexPoly(r_map, np.int32(facet), int(i))

        return l_map, r_map



class OmmatidiaGauss(OmmatidiaFeatures):
    """
    OmmatidiaGauss prepcrocessing class.
    Tranforms every fisheye (circular) image to a set of features.
    This class is an extension of the OmmatidiaFeatures class which gets the
    colour-values of the ommatidia using their original area setted using
    Gaussian distributions.
    """

    # the pattern of the maps' file path.
    # for every model we compute a different pair of maps
    MapsFilenamePattern = filepath + cparams['gaus']['filepatt']


    def __init__(self,
                model=SmolkaModel(OmmatidiaFeatures.FRadius), bw=False,
                colour_spec=True, fisheye=True, rebuild_map=False):
        """
        Initialises the featrures of the ommatidia.

        :param model:       the subsampling model for the feature extraction
        :param fisheye:     whether the input images represent fisheye shapes
        :param rebuild_map: if you want to rebuild the map
        """
        super(OmmatidiaGauss, self).__init__(model, fisheye=fisheye, bw=bw,
                        colour_spec=colour_spec, rebuild_kernels=rebuild_map)

        filename = OmmatidiaGauss.MapsFilenamePattern % model.__class__.Name
        if os.path.exists(filename) and not rebuild_map:
            self.lmap, self.rmap = pickle.load( open( filename, "rb" ) )
        elif not os.path.isdir(filepath + "MAP") and not rebuild_map:
            os.makedirs(filepath + "MAP")

        if not os.path.exists(filename) or rebuild_map:
            self.SaveMap(filename)


    def preprocess(self, data, return_img=False, raw=False):
        """
        Tranforms every fisheye (circular) image to a ommatidia values.

        Input:

        :param data:          the input data
        :param return_img:    if true, also return the eyes images as output
        :param raw:           if false, returns the image produced by the get_values
                              function

        Output:

        :return data:          the raw data
        :return lcolour:       the colours of the ommatidia of the left eye
        :return rcolour:       the colours of the ommatidia of the right eye

        Optional:

        :return leye:          the image of the left eye
        :return reye:          the image of the right eye
        """
        data, _, _, leye, reye = \
            super(OmmatidiaGauss, self).preprocess(data, return_img=True)

        lcolour = np.tensordot(self.lmap, leye, axes=([1,2], [0,1]))
        rcolour = np.tensordot(self.rmap, reye, axes=([1,2], [0,1]))

        if return_img:
            if raw:
                return (data, lcolour, rcolour, leye, reye)
            else:
                return (data, lcolour, rcolour,
                    self.get_values(leye, lcolour, self.l_sph),
                    self.get_values(reye, rcolour, self.r_sph))
        else:
            return (data, lcolour, rcolour)


    def BuildMap(self, Ws=int(Scale*Width), Hs=int(Scale*Height), maxlen=None):
        """
        Builds the maps (x and y) using the given parameters.

        :param Ws: source width
        :param Hs: source height
        """
        l_feats, r_feats = self.model.get_ommatidia_gaussians()

        G_x, G_y = np.mgrid[0:Ws, 0:Hs]
        G_pos = np.empty(G_x.shape + (2,))
        G_pos[:, :, 0] = G_x
        G_pos[:, :, 1] = G_y

        depth = maxlen if maxlen else np.min([l_feats.shape[1], r_feats.shape[1]])
        lmap = np.zeros((depth, Hs, Ws), np.float32)
        rmap = np.zeros((depth, Hs, Ws), np.float32)

        for i in range(depth):
            mu_x, mu_y, C_11, C_12, C_21, C_22, theta, phi, s = l_feats[:,i]
            mu = np.asarray([int(mu_x+Ws/2), int(-mu_y+Hs/2)])
            C = OmmatidiaFeatures.CScale * np.asarray([[C_11, C_12], [C_21, C_22]])
            lG = mvgauss(mu, C)

            mu_x, mu_y, C_11, C_12, C_21, C_22, theta, phi, s = r_feats[:,i]
            mu = np.asarray([int(mu_x+Ws/2), int(-mu_y+Hs/2)])
            C = OmmatidiaFeatures.CScale * np.asarray([[C_11, C_12], [C_21, C_22]])
            rG = mvgauss(mu, C)

            lmap[i,:,:] = lG.pdf(G_pos)
            rmap[i,:,:] = rG.pdf(G_pos)

            lmap[i,:,:] = lmap[i,:,:] / np.sum(lmap[i,:,:])
            rmap[i,:,:] = rmap[i,:,:] / np.sum(rmap[i,:,:])

        return lmap, rmap
