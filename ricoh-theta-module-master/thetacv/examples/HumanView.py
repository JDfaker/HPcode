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
# ------- Script Description ---------------------------------------------------
# Camera perceives the world through the Crabs optical characteristics
# This script is used to test the live camera preview after feature extraction
# and voronoi diagram representation.


from thetacv.capture.stream import Frame, Camera
from thetacv.capture.preprocess import NonePreprocess, Dewarp, Panorama

import sys, getopt
import cv2, time
import numpy as np


class ExampleHumanView(object):
    """
    """

    # default previewed image size
    img_size = (1024, 512)


    def __init__(self, smolka = True, sim = False):

        # Initialise the preprocessor
        #
        # preprocessor = NonePreprocess(colour_spec=False, bw=False)
        preprocessor = Dewarp(colour_spec=False, bw=False, rebuild_map=True)

        #preprocessor = Panorama(colour_spec=False, bw=False, rebuild_map=True)

        # Initialise the stream of the camera with the appropriate preprocessor
        self.stream = Camera(preprocessor=preprocessor)


    def showView(self, imgsize = img_size):

        # receive images till stopped
        while True:
            # measure time
            stime = time.time()
            # get the new frame
            frame = self.stream.next()

            # print the frame rate
            print " Perception Frame Rate is : ", 1. / (time.time() - stime), \
                                                                        "FPS"

            # place side by side the left and right eye perceived images
            img = np.concatenate((frame.leye, frame.reye), axis=1)

            # read actual image size
            original_img_size = np.asarray(img.shape[:2][::-1],\
                                                            dtype=np.float32)

            # compute the ratio between the real image and the previewed
            fx, fy = np.asarray(imgsize).astype(float) / \
                                                original_img_size.astype(float)

            # show image
            cv2.imshow('Live', cv2.resize(img, (0, 0), fx=fx, fy=fy))

            # wait 1 milliseconds till the preview its updated
            cv2.waitKey(1)





# -----------------------------------------------------------------------------#
""" Argument handler """
def main(argv):
   view = ''
   model = ''
   w = ''
   h = ''
   try:
      opts, args = getopt.getopt(argv,"sv:m:w:h:")
   except getopt.GetoptError:
      print 'CrabsView.py -v <ViewName> -m <ModelName> -w <ImageWidth> \
                                                        -h <ImageHeight>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-s':
         print 'CrabsView.py -v <ViewName> -m <ModelName> -w <ImageWidth> \
                                                          -h <ImageHeight>'
         print "ViewName can be: "
         print "               i) RealWorld  "
         print "               ii) Simulation  "
         print "ModelName can be: "
         print "               i) Smolka  "
         print "               ii) Custom  "
         print "ImageWidth has to be positive"
         print "ImageHeight has to be positive"

         sys.exit()
      elif opt == "-v":
         view = arg
      elif opt == "-m":
         model = arg
      elif opt == "-w":
         w = arg
      elif opt == "-h":
         h = arg

   if view == '':
        print 'Please use the CrabsView.py -s'
   else:
       if w == '' or h == '':
           print 'ViewName is ', view , "with model ", model, \
           "and preview image size is ", "(1024,512)"
       else:
           print 'ViewName is ', view , "with model ", model, \
           "and preview image size is ", "(",w,",",h,")"

       if (view == "RealWorld"):
           if (model == "Smolka"):
               ecv = ExampleCrabsView()
           elif (model == "Custom"):
               ecv = ExampleCrabsView()
           else:
               print "Wrong Model name was provided"
               sys.exit(2)

       elif (view == "Simulation"):
            if (model == "Smolka"):
                ecv = ExampleCrabsView(sim = True)
            elif (model == "Custom"):
                ecv = ExampleCrabsView(sim = True)
            else:
                print "Wrong Model name was provided"
                sys.exit(2)
       else:
           print "Wrong ViewName name was provided"
           sys.exit(2)

       if w == '' or h == '':
           ecv.showView()
       else:
           ecv.showView( imgsize = (int(w), int(h)) )

       # del ecv class
       del ecv




if __name__ == '__main__':

    #main(sys.argv[1:])
    ehv = ExampleHumanView()

    ehv.showView()
