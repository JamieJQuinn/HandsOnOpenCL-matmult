# HandsOnOpenCL-matmult

Solutions to the [HandsOnOpenCL](https://github.com/HandsOnOpenCL/Lecture-Slides/releases/tag/v1.2) matrix multiplication exercises working on OpenCL 2.0+ for C++

You will require the C++ opencl headers to be available on your machine (packages `opencl-headers` and `opencl-clhpp` on Arch Linux) and a modern gcc compiler capable of compiling C++17 (not necessary for OpenCL, just some useful host-side code). These can be installed by most package managers. You will also need some kind of OpenCL runtime. I have tested this with

- Intel's GPU runtime (`intel-opencl-runtime` on Arch)
- Intel's CPU runtime (`intel-compute-runtime` on Arch)

To run, change the line in `main` in `host.cpp` (L330 or so) to search for the OpenCL platform you want to target, type `make` and run the output executable `exe`. This will print out a list of available OpenCL platforms then run a series of different kernels on the target, along with a sequential CPU implementation and an OpenMP CPU implementation.
