SCRIPTBOTS
Author: Andrej Karpathy
Contributors: Casey Link
Licence: BSD

Project website and attached forum: 
https://sites.google.com/site/scriptbotsevo/home

------------------------------------------------------

BUILDING:
To compile scriptbots you will need:

CMake >= 2.8 (http://www.cmake.org/cmake/resources/software.html)
OpenGL and GLUT (http://www.opengl.org/resources/libraries/glut/)
* Linux: freeglut (http://freeglut.sourceforge.net/)
It will use OpenMP to speed up everything, in case you have multicore cpu.

If you are on Ubuntu (or debian) you can install all the dependencies with:
$ apt-get install cmake build-essential libopenmpi-dev libglut32-dev libxi-dev libxmu-dev

To build ScriptBots on Linux:
$ cd path/to/source
$ mkdir build
$ cd build
$ cmake ../ # this is the equiv of ./configure
$ make

To execute ScriptBots simply type the following in the build directory:
$ ./scriptbots

For Windows:
Follow basically the same steps, but after running cmake open up the VS solution (.sln) file it generates and compile the project from VS.

USAGE:

Compile and run. Hit 'd' to make it go MUCH FASTER (and not draw). Speed can
also be controlled with + and -. Use mouse to pan and zoom around.


QUESTIONS COMMENTS are best posted at the google group, available on project site
or contact me at andrej.karpathy@gmail.com

