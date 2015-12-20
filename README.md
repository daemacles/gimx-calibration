# About

Calibrate the mouse input control curves for GIMX using a video feed.

# Usage

## Requirements

  * daemacles/cpp-matplotlib
  * Eigen3
  * OpenCV
  * FlyCap2 SDK from PointGrey

## Compiling

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ ./gimx-calibration

By default cmake is set to use clang, edit CMakeLists.txt if you need
something else.
