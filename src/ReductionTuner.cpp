// Copyright 2016 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <vector>
#include <iostream>
#include <exception>
#include <algorithm>
#include <iomanip>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <Reduction.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

const inputDataType magicValue = 42;

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * input, cl::Buffer * input_d, cl::Buffer * output_d, const unsigned int outputSize);

int main(int argc, char * argv[]) {
  unsigned int nrIterations = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  unsigned int vectorSize = 0;
  unsigned int maxThreads = 0;
  unsigned int maxItems = 0;
  unsigned int inputSize = 0;
  TuneBench::reductionConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    vectorSize = args.getSwitchArgument< unsigned int >("-vector");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    inputSize = args.getSwitchArgument< unsigned int >("-input_size");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -vector ... -max_threads ... -max_items ... -input_size ... -output_size ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  cl::Context clContext;
  std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
  std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
  std::vector< std::vector< cl::CommandQueue > > * clQueues = 0;

  // Allocate host memory
  std::vector< inputDataType > input(inputSize);
  cl::Buffer input_d, output_d;

  std::fill(input.begin(), input.end(), magicValue);

  std::cout << std::fixed << std::endl;
  std::cout << "# inputSize outputSize nrItemsPerBlock nrThreadsD0 nrItemsD0 GB/s time stdDeviation COV" << std::endl << std::endl;

  for ( unsigned int threads = vectorSize; threads <= maxThreads; threads += vectorSize ) {
    conf.setNrThreadsD0(threads);
    for ( unsigned int items = 1; items <= maxItems; items++ ) {
      conf.setNrItemsD0(items);
      if ( inputSize % (conf.getNrThreadsD0() * conf.getNrItemsD0()) != 0 ) {
        continue;
      }
      for ( unsigned int itemsPerBlock = 1; itemsPerBlock < inputSize; itemsPerBlock++ ) {
        conf.setNrItemsPerBlock(itemsPerBlock * (conf.getNrThreadsD0() * conf.getNrItemsD0()));
        if ( conf.getItemsPerBlock() > inputSize ) {
          break;
        } else if ( inputSize % conf.getNrItemsPerBlock() != 0 ) {
          continue;
        }

        // Generate kernel
        unsigned int outputSize = inputSize / conf.getNrItemsPerBlock();
        double gbytes = isa::utils::giga((static_cast< uint64_t >(inputSize) * sizeof(inputDataType)) + (static_cast< uint64_t >(outputSize) * sizeof(outputDataType)));
        std::vector< outputDataType > output(outputSize);
        cl::Event clEvent;
        cl::Kernel * kernel;
        isa::utils::Timer timer;
        std::string * code = TuneBench::getReductionOpenCL(conf, inputDataName);

        delete clQueues;
        clQueues = new std::vector< std::vector< cl::CommandQueue > >();
        isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
        try {
          initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &input, &input_d, &output_d, outputSize);
        } catch ( cl::Error & err ) {
          return -1;
        }
        try {
          kernel = isa::OpenCL::compile("reduction", *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
        } catch ( cl::Error & err ) {
          std::cerr << err.what() << std::endl;
          delete code;
          break;
        }
        delete code;

        cl::NDRange global(inputSize / conf.getNrItemsPerBlock());
        cl::NDRange local(conf.getNrThreadsD0());

        kernel->setArg(0, input_d);
        kernel->setArg(1, output_d);

        try {
          // Warm-up run
          clQueues->at(clDeviceID)[0].finish();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &clEvent);
          clEvent.wait();
          // Tuning runs
          for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
            timer.start();
            clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &clEvent);
            clEvent.wait();
            timer.stop();
          }
          clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(outputDataType), reinterpret_cast< void * >(output.data()), 0, &clEvent);
          clEvent.wait();
        } catch ( cl::Error & err ) {
          std::cerr << "OpenCL kernel execution error (";
          std::cerr << conf.print();
          std::cerr << "), (";
          std::cerr << isa::utils::toString(inputSize / conf.getNrItemsPerBlock()) << "): ";
          std::cerr << isa::utils::toString(err.err()) << std::endl;
          delete kernel;
          if ( err.err() == -4 || err.err() == -61 ) {
            return -1;
          }
          break;
        }
        delete kernel;

        bool error = false;
        for ( auto item = output.begin(); item != output.end(); ++item ) {
          if ( *item != (magicValue * conf.getNrItemsPerBlock()) ) {
            std::cerr << "Output error (" << conf.print() << ")." << std::endl;
            error = true;
            break;
          }
        }
        if ( error ) {
          continue;
        }

        std::cout << inputSize << " ";
        std::cout << conf.print() << " ";
        std::cout << std::setprecision(3);
        std::cout << gbytes / timer.getAverageTime() << " ";
        std::cout << std::setprecision(6);
        std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
      }
    }
  }
  std::cout << std::endl;

  return 0;
}

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * input, cl::Buffer * input_d, cl::Buffer * output_d, const unsigned int outputSize) {
  try {
    *input_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, input->size() * sizeof(inputDataType), 0, 0);
    *output_d = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, outputSize * sizeof(outputDataType), 0, 0);
    clQueue->enqueueWriteBuffer(*input_d, CL_FALSE, 0, input->size() * sizeof(inputDataType), reinterpret_cast< void * >(input->data()));
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error (memory initialization): " << isa::utils::toString(err.err()) << "." << std::endl;
    throw;
  }
}

