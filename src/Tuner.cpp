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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <ArgumentList.hpp>

#include <BlackScholes.hpp>
#include <Reduction.hpp>
#include <Correlator.hpp>
#include <MD.hpp>
#include <Stencil.hpp>
#include <Triad.hpp>
#include <StringToArgcArgv.hpp>

int runKernel(std::string kernel, int argc, char * argv[]);
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

bool fileExists(std::string filename) {
  std::ifstream file(filename);
  return file.good();
}

int main(int argc, char * argv[]) {
  std::cout << "Starting kernels" << std::endl << std::endl;

  int BlackScholesResult = -1;
  int CorrelatorResult = -1;
  int MDResult = -1;
  int ReductionResult = -1;
  int StencilResult = -1;
  int TriadResult = -1;

  try {
    isa::utils::ArgumentList args(argc, argv);

    if(args.getSwitch("-file_input")) {
      std::string file = args.getSwitchArgument<std::string>("-file");
      if(fileExists(file)) {
        std::cout << "Reading settings from file: " << file << std::endl;
        std::ifstream fileStream(file);
        std::string line;

        for( std::string line; getline( fileStream, line ); )
        {
          std::string kernel = line.substr(0, line.find(' '));
          std::string str = line.substr(line.find(' '));
          int argcK;
          char **argvK;

          stringToArgcArgv(argv[0] + str, &argcK, &argvK);

          int result = runKernel(kernel, argcK, argvK);
          if(kernel == "blackscholes") {
            BlackScholesResult = result;
          } else if(kernel == "correlator") {
            CorrelatorResult = result;
          } else if(kernel == "md") {
            MDResult = result;
          } else if(kernel == "reduction") {
            ReductionResult = result;
          } else if(kernel == "stencil") {
            StencilResult = result;
          } else if(kernel == "triad") {
            TriadResult = result;
          }
        }

      } else {
        std::cerr << "Can\'t read file, please check path" << std::endl;
        return 1;
      }
    } else {
      BlackScholesResult = runKernel("blackscholes", argc, argv);
      CorrelatorResult = runKernel("correlator", argc, argv);
      MDResult = runKernel("md", argc, argv);
      ReductionResult = runKernel("reduction", argc, argv);
      StencilResult = runKernel("stencil", argc, argv);
      TriadResult = runKernel("triad", argc, argv);
    }
  } catch (isa::utils::EmptyCommandLine & err) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "Method 1: " << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -input_size ... -max_threads ... --loop_unrolling ..." << std::endl;
    std::cerr << "Method 2:" << argv[0] << " -file_input -file ..." << std::endl;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

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

int runKernel(std::string kernel, int argc, char * argv[]) {
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

  if(kernel == "blackscholes") {
    try {
      isa::utils::ArgumentList args(argc, argv);

      clPlatformID = args.getSwitchArgument < unsigned int > ("-opencl_platform");
      clDeviceID = args.getSwitchArgument < unsigned int > ("-opencl_device");
      nrIterations = args.getSwitchArgument < unsigned int > ("-iterations");
      inputSize = args.getSwitchArgument < unsigned int > ("-input_size");
      maxThreads = args.getSwitchArgument < unsigned int > ("-max_threads");
      loopUnrolling = args.getSwitchArgument < unsigned int > ("-loop_unrolling");
    } catch (isa::utils::EmptyCommandLine &err) {
      std::cerr << argv[0]
                << " -opencl_platform ... -opencl_device ... -iterations ... -input_size ... -max_threads ... --loop_unrolling ..."
                << std::endl;
      return 1;
    } catch (std::exception &err) {
      std::cerr << err.what() << std::endl;
      return 1;
    }

    /* Execute kernels */
    std::cout << "BlackScholes" << std::endl;
    return BlackScholes::runKernel(clPlatformID, clDeviceID, nrIterations, inputSize, maxThreads, loopUnrolling);
  } else if(kernel == "correlator") {
    try {
      isa::utils::ArgumentList args(argc, argv);

      clPlatformID = args.getSwitchArgument < unsigned
      int > ("-opencl_platform");
      clDeviceID = args.getSwitchArgument < unsigned
      int > ("-opencl_device");
      padding = args.getSwitchArgument < unsigned
      int > ("-padding");
      nrIterations = args.getSwitchArgument < unsigned
      int > ("-iterations");
      vectorSize = args.getSwitchArgument < unsigned
      int > ("-vector");
      maxThreads = args.getSwitchArgument < unsigned
      int > ("-max_threads");
      maxItems = args.getSwitchArgument < unsigned
      int > ("-max_items");
      maxUnroll = args.getSwitchArgument < unsigned
      int > ("-max_unroll");
      sequentialTime = args.getSwitch("-sequential_time");
      parallelTime = args.getSwitch("-parallel_time");
      cellWidth = args.getSwitchArgument < unsigned
      int > ("-width");
      cellHeight = args.getSwitchArgument < unsigned
      int > ("-height");
      nrChannels = args.getSwitchArgument < unsigned
      int > ("-channels");
      nrStations = args.getSwitchArgument < unsigned
      int > ("-stations");
      nrSamples = args.getSwitchArgument < unsigned
      int > ("-samples");
    } catch (isa::utils::EmptyCommandLine &err) {
      std::cerr << argv[0]
                << " -opencl_platform ... -opencl_device ... -padding ... -iterations ... -vector ... -max_threads ... -max_items ... -max_unroll ... [-sequential_time | -parallel_time] -width ... -height ... -channels ... -stations ... -samples ..."
                << std::endl;
      return 1;
    } catch (std::exception &err) {
      std::cerr << err.what() << std::endl;
      return 1;
    }

    std::cout << std::endl << "Correlator" << std::endl;
    return Correlator::runKernel(clPlatformID, clDeviceID, padding, nrIterations, vectorSize, maxThreads, maxItems,
                                 maxUnroll, sequentialTime, parallelTime, cellWidth, cellHeight, nrChannels, nrStations,
                                 nrSamples);
  } else if(kernel == "md") {
    try {
      isa::utils::ArgumentList args(argc, argv);

      clPlatformID = args.getSwitchArgument < unsigned
      int > ("-opencl_platform");
      clDeviceID = args.getSwitchArgument < unsigned
      int > ("-opencl_device");
      nrIterations = args.getSwitchArgument < unsigned
      int > ("-iterations");
      vectorSize = args.getSwitchArgument < unsigned
      int > ("-vector");
      maxThreads = args.getSwitchArgument < unsigned
      int > ("-max_threads");
      maxItems = args.getSwitchArgument < unsigned
      int > ("-max_items");
      nrAtoms = args.getSwitchArgument < unsigned
      int > ("-atoms");
    } catch (isa::utils::EmptyCommandLine &err) {
      std::cerr << argv[0]
                << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -atoms ..."
                << std::endl;
      return 1;
    } catch (std::exception &err) {
      std::cerr << err.what() << std::endl;
      return 1;
    }

    std::cout << std::endl << "MD" << std::endl;
    return MD::runKernel(clPlatformID, clDeviceID, nrIterations, vectorSize, maxThreads, maxItems, nrAtoms);
  } else if(kernel ==  "reduction") {
    try {
      isa::utils::ArgumentList args(argc, argv);

      clPlatformID = args.getSwitchArgument < unsigned
      int > ("-opencl_platform");
      clDeviceID = args.getSwitchArgument < unsigned
      int > ("-opencl_device");
      nrIterations = args.getSwitchArgument < unsigned
      int > ("-iterations");
      vectorSize = args.getSwitchArgument < unsigned
      int > ("-vector");
      maxThreads = args.getSwitchArgument < unsigned
      int > ("-max_threads");
      maxItems = args.getSwitchArgument < unsigned
      int > ("-max_items");
      maxVector = args.getSwitchArgument < unsigned
      int > ("-max_vector");
      inputSize = args.getSwitchArgument < unsigned
      int > ("-input_size");
    } catch (isa::utils::EmptyCommandLine &err) {
      std::cerr << argv[0]
                << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -max_vector ... -input_size ... "
                << std::endl;
      return 1;
    } catch (isa::utils::SwitchNotFound &err) {
      std::cerr << argv[0]
                << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -max_vector ... -input_size ... "
                << std::endl;
      std::cerr << err.what() << std::endl;
      return 1;
    } catch (std::exception &err) {
      std::cerr << "Exception" << std::endl;
      std::cerr << err.what() << std::endl;
      return 1;
    }

    std::cout << std::endl << "Reduction" << std::endl;
    return Reduction::runKernel(clPlatformID, clDeviceID, nrIterations, vectorSize, maxThreads, maxItems, maxVector,
                                inputSize);
  } else if (kernel == "triad") {
    try {
      isa::utils::ArgumentList args(argc, argv);

      clPlatformID = args.getSwitchArgument < unsigned
      int > ("-opencl_platform");
      clDeviceID = args.getSwitchArgument < unsigned
      int > ("-opencl_device");
      nrIterations = args.getSwitchArgument < unsigned
      int > ("-iterations");
      vectorSize = args.getSwitchArgument < unsigned
      int > ("-vector");
      maxThreads = args.getSwitchArgument < unsigned
      int > ("-max_threads");
      maxItems = args.getSwitchArgument < unsigned
      int > ("-max_items");
      maxVector = args.getSwitchArgument < unsigned
      int > ("-max_vector");
      inputSize = args.getSwitchArgument < unsigned
      int > ("-input_size");
    } catch (isa::utils::EmptyCommandLine &err) {
      std::cerr << argv[0]
                << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -max_vector ... -input_size ... "
                << std::endl;
      return 1;
    } catch (std::exception &err) {
      std::cerr << err.what() << std::endl;
      return 1;
    }

    std::cout << std::endl << "Triad" << std::endl;
    return Triad::runKernel(clPlatformID, clDeviceID, nrIterations, vectorSize, maxThreads, maxItems, maxVector,
                            inputSize);
  } else if(kernel == "stencil") {
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
      return Stencil::runKernel(clPlatformID, clDeviceID, nrIterations, vectorSize, padding, maxThreads, maxItems, matrixWidth, localMemory);
  }

  return -1;
}