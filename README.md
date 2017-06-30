
# TuneBench

Simple tunable OpenCL kernels for many-core accelerators.
The goal of this collection of kernels is benchmarking how much tuning affects performance on a variety of many-core platforms, therefore the kernels are not feature complete.

## Dependencies

* [OpenCL](https://github.com/isazi/OpenCL) - master branch
* [utils](https://github.com/isazi/utils) - master branch

## Run kernels
To run the kernels, either provide all parameters using the commandline:
```
./bin/Tuner -opencl_platform 0 -opencl_device 0 -iterations 10 -input_size 1000 -max_threads 1024 -loop_unrolling 2 -padding 2 -vector 2 -max_items 100 -max_unroll 2 -parallel_time -width 2 -height 2 -channels 4 -stations 4 -samples 4 -atoms 10 -max_vector 8 -matrix_width 2
```

The other option is to use a file as input, use the following as input parameters:

```
./bin/tuner -file_input -file [filename]
```

Example file:
```
blackscholes -opencl_platform 0 -opencl_device 0 -iterations 10 -input_size 1000 -max_threads 1024 -loop_unrolling 2
correlator -opencl_platform 0 -opencl_device 0 -iterations 10 -padding 2 -vector 2 -max_threads 1024 -max_items 100 -max_unroll 2 -parallel_time -width 2 -height 2 -channels 4 -stations 4 -samples 4
md -opencl_platform 0 -opencl_device 0 -iterations 10 -vector 2 -max_threads 1024 -max_items 100 -atoms 10
reduction -opencl_platform 0 -opencl_device 0 -iterations 10 -vector 2 -max_threads 1024 -max_items 100 -max_vector 8 -input_size 1000
stencil -opencl_platform 0 -opencl_device 0 -iterations 10 -vector 2 -padding 2 -max_threads 1024 -max_items 100 -matrix_width 2
triad -opencl_platform 0 -opencl_device 0 -iterations 10 -vector 2 -max_threads 1024 -max_items 100 -max_vector 8 -input_size 1000
```

## License

Licensed under the Apache License, Version 2.0.

