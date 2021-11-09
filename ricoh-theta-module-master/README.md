# ThetaCV
Description:
This project provides the interface to Ricoh Theta S camera utilising OpenCV
and python. Additionally, a number of different image preproccesors are
available to transform the raw fisheye images received from the camera to
image space as perceived from human and other animals optical properties.
Currently, the only animal supported is the fiddler crabs.

# Installation

## OpenCV dependences

We suggest you make sure that you have installed all the dependences of the OpenCV library
before you proceed with the installation.
You can do this by running the following commands:

`sudo apt-get update`  
`sudo apt-get upgrade`  
`sudo apt-get install build-essential cmake git pkg-config`  
`sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev`  
`sudo apt-get install libgtk2.0-dev`  
`sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev`  
`sudo apt-get install libatlas-base-dev gfortran`  

## Install virtual environment

In the repository's root directory run the command:  
`source init_activate.sh`  
to create the virtual environment and activate it.  

This virtual enviromnent has to be activated every time before you run your
examples. This can be achived by running the command:  
`theta_venv`  
If you want to deactivate the virtual environment you just run the command:  
`deactivate`  


## Installing the ThetaCV package

To install the package you need to go to the root directory and run:  

`make`  
`make install`  
  
## Uninstall the ThetaCV package

`pip uninstall thetacv`  


## Contributors
University of Edinburgh  

Evripidis Gkanias, email: ev.gkanias@gmail.com  

Theodoros Stouraitis, email:  stoutheo@gmail.com  

## License

The license used is MIT.  
