// Copyright 2017 Robin Hansma <robin.hansma@student.uva.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>

#include <ArgumentList.hpp>

#include <BlackScholes.hpp>
#include <Reduction.hpp>
#include <Correlator.hpp>
#include <MD.hpp>
#include <Stencil.hpp>
#include <Triad.hpp>

/**
 * Check result of the kernel, non-zero value means error occured
 * @param result    return code of kernel
 * @param kernel    name of the kernel
 */
void checkResult(int result, std::string kernel) {
  if(result == 0) {
    std::cout << "Successfully executed " << kernel << " Kernel" << std::endl;
  } else {
    std::cerr << "Error while executing " << kernel << " Kernel" << std::endl;
  }
}

int main(int argc, char * argv[]) {
  std::cout << "Starting kernels" << std::endl << std::endl;
  unsigned int nrIterations = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  unsigned int maxThreads = 0;
  unsigned int inputSize = 0;
  unsigned int loopUnrolling = 0;
  unsigned int padding = 0;
  unsigned int vectorSize = 0;
  unsigned int maxItems = 0;
  unsigned int maxUnroll = 0;
  unsigned int nrChannels = 0;
  unsigned int nrStations = 0;
  unsigned int nrSamples = 0;
  unsigned int cellWidth = 0;
  unsigned int cellHeight = 0;
  unsigned int nrAtoms = 0;
  unsigned int maxVector = 0;
  unsigned int matrixWidth = 0;
  bool parallelTime = false;
  bool sequentialTime = false;
  bool localMemory = false;

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    inputSize = args.getSwitchArgument< unsigned int >("-input_size");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    loopUnrolling = args.getSwitchArgument< unsigned int >("-loop_unrolling");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -input_size ... -max_threads ... --loop_unrolling ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  /* Execute kernels */
  std::cout << "BlackScholes" << std::endl;
  int BlackScholesResult = BlackScholes::runKernel(clPlatformID, clDeviceID, nrIterations, inputSize, maxThreads, loopUnrolling);

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    vectorSize = args.getSwitchArgument< unsigned int >("-vector");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    maxUnroll = args.getSwitchArgument< unsigned int >("-max_unroll");
    sequentialTime = args.getSwitch("-sequential_time");
    parallelTime = args.getSwitch("-parallel_time");
    cellWidth = args.getSwitchArgument< unsigned int >("-width");
    cellHeight = args.getSwitchArgument< unsigned int >("-height");
    nrChannels = args.getSwitchArgument< unsigned int >("-channels");
    nrStations = args.getSwitchArgument< unsigned int >("-stations");
    nrSamples = args.getSwitchArgument< unsigned int >("-samples");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -padding ... -iterations ... -vector ... -max_threads ... -max_items ... -max_unroll ... [-sequential_time | -parallel_time] -width ... -height ... -channels ... -stations ... -samples ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::cout << std::endl << "Correlator" << std::endl;
  int CorrelatorResult = Correlator::runKernel(clPlatformID, clDeviceID, padding, nrIterations, vectorSize, maxThreads, maxItems, maxUnroll, sequentialTime, parallelTime, cellWidth, cellHeight, nrChannels, nrStations, nrSamples);

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    vectorSize = args.getSwitchArgument< unsigned int >("-vector");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    nrAtoms = args.getSwitchArgument< unsigned int >("-atoms");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -atoms ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::cout << std::endl << "MD" << std::endl;
  int MDResult = MD::runKernel(clPlatformID, clDeviceID, nrIterations, vectorSize, maxThreads, maxItems, nrAtoms);

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    vectorSize = args.getSwitchArgument< unsigned int >("-vector");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    maxVector = args.getSwitchArgument< unsigned int >("-max_vector");
    inputSize = args.getSwitchArgument< unsigned int >("-input_size");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -max_vector ... -input_size ... " << std::endl;
    return 1;
  } catch ( isa::utils::SwitchNotFound & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -max_vector ... -input_size ... " << std::endl;
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << "Exception" << std::endl;
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::cout << std::endl << "Reduction" << std::endl;
  int ReductionResult = Reduction::runKernel(clPlatformID, clDeviceID, nrIterations, vectorSize, maxThreads, maxItems, maxVector, inputSize);

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    vectorSize = args.getSwitchArgument< unsigned int >("-vector");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    matrixWidth = args.getSwitchArgument< unsigned int >("-matrix_width");
    localMemory = args.getSwitch("-local");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -padding ... -max_threads ... -max_items ... -matrix_width ... [-local]" << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::cout << std::endl << "Stencil" << std::endl;
  int StencilResult = Stencil::runKernel(clPlatformID, clDeviceID, nrIterations, vectorSize, padding, maxThreads, maxItems, matrixWidth, localMemory);

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    vectorSize = args.getSwitchArgument< unsigned int >("-vector");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    maxVector = args.getSwitchArgument< unsigned int >("-max_vector");
    inputSize = args.getSwitchArgument< unsigned int >("-input_size");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -max_vector ... -input_size ... " << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::cout << std::endl << "Triad" << std::endl;
  int TriadResult = Triad::runKernel(clPlatformID, clDeviceID, nrIterations, vectorSize, maxThreads, maxItems, maxVector, inputSize);

  std::cout << std::endl;

  /* Check results */
  std::cout << "Summary: " << std::endl;
  checkResult(BlackScholesResult, "BlackScholes");
  checkResult(CorrelatorResult, "Correlator");
  checkResult(MDResult, "MD");
  checkResult(ReductionResult, "Reduction");
  checkResult(StencilResult, "Stencil");
  checkResult(TriadResult, "Triad");

  return 0;
}