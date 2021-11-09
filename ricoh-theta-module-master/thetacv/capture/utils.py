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

import os
import cv2
import numpy as np


def RotBlur(image, phi):
    """
    Return a Rotationally blurred image of the input image.
    Used to perform a GaussianBlur on a circular ring with
    skewed Gaussians.

    :image:   the uprocessed image of the eye
    :phi:     the angle phi is utilised for the rotational blur
    """

    rows, cols , dep = image.shape
    # generate duplicate
    img_temp = image.copy()

    # rotate left for phi degrees
    M = cv2.getRotationMatrix2D( (cols * .5, rows * .5), phi , 1)
    lrot = cv2.warpAffine(image, M, (cols, rows))

    # rotate right for a degrees
    M2 = cv2.getRotationMatrix2D((cols * .5, rows * .5), -phi, 1)
    rrot = cv2.warpAffine(image, M2, (cols, rows))

    # blend the three images into one
    mix_l = cv2.addWeighted(image, .33, lrot, .33, 0)
    mix_l_r = cv2.addWeighted(mix_l, 0.67, rrot, .33, 0)

    return mix_l_r      # return the rotationally blurred image


def _vis_G_kernels(un_kernel2D_3c, inv_un_kernel2D_3c):
    """
    Visualising the Kernels

    :un_kernel2D_3c:       Kernel A
    :inv_un_kernel2D_3c:   Kernel B
    """

    cv2.imshow('kernel A', un_kernel2D_3c)
    cv2.imshow('kernel B', inv_un_kernel2D_3c)
    cv2.waitKey()

def mix_kernels(w_kernel, h_kernel, channels, sigma, show = False):
    """
    Return a tuple with the two kernels.
    Kernel :un_K2D_3c: is a 2D Gaussian Kernel the size of
    the kernelSize**2
    Kernel :inv_un_K2D_3c: is a 2D Inverse Gaussian Kernel
    the size of the kernelSize**2

    :kernelSize:     the size of the 1D Gaussian Kernel
    :sigma:          variance of the 1D Gaussian
    """

    # generate Gaussian kernel 1D
    K1D_1 = cv2.getGaussianKernel(w_kernel, sigma)
    K1D_2 = cv2.getGaussianKernel(h_kernel, sigma)

    # generate Gaussian kernel 2D
    Ker2D = K1D_1 * K1D_2.T
    
    # normalise the Gaussian 2D kernel in the centre
    un_K2D = (Ker2D / (Ker2D[ w_kernel // 2, h_kernel // 2 ]))

    # Gaussian mask for the A image(GaussianBlurred) in all 3 channels
    un_K2D_3c = np.concatenate([un_K2D[...,np.newaxis]] * channels, axis=2)

    # inverse Gaussian mask for the B image (Rotated) in all 3 channels
    inv_un_K2D_3c = 1. - un_K2D_3c

    if show:
        _vis_G_Ks(un_K2D_3c, inv_un_K2D_3c)

    return (un_K2D_3c, inv_un_K2D_3c)


def blend_mix_imgs(img_A, kernel_A, img_B, kernel_B, mix_coef=.5, debug=False):
    """
    Return a tuple with the blended image and the two images after
    the respective kernels have been applied (Debugging).

    :img_A:      image A (GaussianBlurred)
    :kernel_A:   2D Gaussian Kernel the size of the image A
    :img_B:      image B (RotationalBlurred)
    :kernel_A:   2D Inverse Gaussian Kernel the size of the image B
    :mix_coef:   scale value defining the blending coefficient
    """

    # apply Gaussian mask in the image
    # / 255. normalisation of the kernel w.r.t to color scale
    img_ker_A = img_A * kernel_A # / 255.

    # apply the inverse Gaussian mask in the image
    # / 255. normalisation of the kernel w.r.t to color scale
    img_ker_B = img_B * kernel_B # / 255.

    # generate the blended image
    # division with mix_coef is used to restore the Initial overall brightness of the image
    #bl_mix_img = cv2.addWeighted(img_ker_A, mix_coef, img_ker_B, 1 - mix_coef, 0) * 2 # mix_coef
    #bl_mix_img = mix_coef * (img_ker_A / 0.5) + (1 - mix_coef) * (img_ker_B / 0.5)
    bl_mix_img = img_ker_A  + img_ker_B

    if debug:
        return (bl_mix_img, img_ker_A, img_ker_B) # blended mixed image and A and B images
    else:
        return bl_mix_img
