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
# This script converts the dataset of the matlab format provided by Jan Hemi
# to a more compact representation for efficient training.
#

import os, logging, logging.config, yaml
import cv2, time, datetime
import numpy as np
import imp

from yaml import BaseLoader
from imutils.video import WebcamVideoStream

from thetacv.capture.preprocess import NonePreprocess
DATASET_MODULE_EXISTS = False


# Loggining configuration
cpath = os.path.dirname(os.path.abspath(__file__))
with open ( cpath + '/logging.yaml', 'rb' ) as config:

    # load dict from yaml file
    yaml_dict = yaml.load(config, Loader=BaseLoader)

    # update a path property to be relative
    yaml_dict['handlers']['info_file']['filename'] =  \
        cpath + '/' + yaml_dict['handlers']['info_file']['filename']

    # update a path property to be relative
    yaml_dict['handlers']['error_file']['filename'] = \
        cpath + '/' + yaml_dict['handlers']['error_file']['filename']

    # logging.config.dictConfig(yaml_dict)
    logger = logging.getLogger(__name__)


class Frame(object):
    """
    Frame object of the 360 camera with dual lences (Default RICOH THETA S).
    """
    Width, Height = 640, 640    # Default width and height of a fisheye image
    _current_frame_id = 1       # A static counter to count the created frames


    def __init__(self, data, preprocessor=NonePreprocess()):
        """
        Creates a frame with a unique ID, the current timestamp and the given
        data. It processes the data using the preprocessor, and takes the new
        data, and the left and right eyes separately. The new data is a back-
        up of the original data.

        :param data:            the raw data provided by the camera
        :param preprocessor:    the preprocessing handler that will transform
                                the input image
        """

        # Frame ID
        self.id = Frame._current_frame_id
        # Current timestamp
        self.timestamp = int(round(time.time() * 1000))
        # Raw data (image), left and right images (eyes) of the camera
        self.data, self.leye, self.reye = preprocessor.preprocess(data)

        Frame._current_frame_id += 1


    def get_id_str(self):
        """
        Returns the frame ID in a string format.
        """
        return str(self.id).zfill(5)


    def get_time_str(self):
        """
        Returns the timestamp in a string format.
        """
        return str(self.timestamp)


    def get_datetime(self):
        """
        Returns the timestamp in a Datetime format.
        """
        return datetime.datetime.fromtimestamp(float(self.timestamp) / 1000)


    def get_dict(self):
        """
        Returns the frame in a dictionary format with entries:
        - Id: string
        - Timestamp: string
        - Data: np.darray
        - Leye: np.darray
        - Reye: np.darray
        """
        return {
            'Id': self.get_id_str(),
            'Timestamp': self.get_time_str(),
            'Data': self.data,
            'Leye': self.leye,
            'Reye': self.reye
        }


    @staticmethod
    def asimage(frame, reye=True, leye=True, data=False,
                    rpath=None, lpath=None, dpath=None):
        """
        Static method which saves the images of the frame.
        If the corresponding file paths are NoneType the timestamp is used as
        the unique name of the each file followed by the first leter of their
        name.

        :param reye:    save the right eye image
        :param leye:    save the left eye image
        :param data:    save the raw data
        :param rpath:   the path of the file for the right eye
        :param lpath:   the path of the file for the left eye
        :param dpath:   the path of the file for the raw data
        """
        if leye:
            if not lpath:
                lpath = "IMG/" + str(frame.timestamp) + "l.png"
            cv2.imwrite(lpath, frame.leye)
        if reye:
            if not rpath:
                rpath = "IMG/" + str(frame.timestamp) + "r.png"
            cv2.imwrite(rpath, frame.reye)
        if data:
            if not dpath:
                dpath = "IMG/" + str(frame.timestamp) + "d.png"
            cv2.imwrite(dpath, frame.data)

        return frame


    @staticmethod
    def fromimage(dpath, rpath=None, lpath=None):
        """
        Static method that loads a frame from saved images.

        :param dpath:   the raw image
        :param rpath:   the right eye image
        :param lpath:   the left eye image
        """
        data = cv2.imread(dpath)
        frame = Frame(cv2.imread(path))
        frame.data = data
        if rpath:
            frame.reye = cv2.imread(rpath)
        if lpath:
            frame.leye = cv2.imread(lpath)
        return frame



class StreamBase(object):
    """
    Interface class for streams of data.
    """

    def __init__(self, preprocessor=NonePreprocess()):
        """
        Initialises and opens the stream.

        :param preprocessor:    the preprocessing method we want to apply on
                                each frame
        """
        self.preprocessor = preprocessor


    def __iter__(self):
        """
        Returns an iterator.
        This allows iterating over the streams.
        """
        return self


    def next(self):
        """
        Returns the next frame.
        """
        return None



class Camera(StreamBase):
    """
    A camera capturer object, that opens, closes and transforms to frames the
    optical flow of the camera.
    """

    def __init__(self, filename=None, preprocessor=NonePreprocess()):
        """
        Opens a connection with the camera.

        :param filename:    the filename of the video we want to load.
                            if this is NoneType, it wont use a file, but the
                            camera stream
        """
        super(Camera, self).__init__(preprocessor=preprocessor)

        if filename:
            # Using a video file to capture from.
            self.video = cv2.VideoCapture(filename)
        else:
            # Using OpenCV to capture from device 0.
            # self.video = cv2.VideoCapture(0)
            i = 1
            while True:
                try:
                    self.video = WebcamVideoStream(src=i).start()
                    break
                except:
                    pass
                i += 1
        # If we decide to use video.mp4, we must have this file in the folder
        # as the main.py.


    def __del__(self):
        """
        Closes the connection with the camera.
        """
        if isinstance(self.video, WebcamVideoStream):
            self.video.stop()
            self.video.stream.release()
        else:
            self.video.release()

    def next(self):
        """
        returns the next available frame of the camera.
        """
        if isinstance(self.video, WebcamVideoStream):
            image = self.video.read()
        else:
            success, image = self.video.read()

        return Frame(image, preprocessor=self.preprocessor)
