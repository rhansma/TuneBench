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
#include <iomanip>
#include <ctime>
#include <algorithm>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <Correlator.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

const unsigned int magicValue = 42;

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * input, cl::Buffer * input_d, const unsigned int outputSize, cl::Buffer * output_d, std::vector< unsigned int > * baselineMap, cl::Buffer * baselineMap_d);

int main(int argc, char * argv[]) {
  bool reInit = true;
  unsigned int padding = 0;
  unsigned int nrIterations = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  unsigned int vectorSize = 0;
  unsigned int maxThreads = 0;
  unsigned int maxItems = 0;
  unsigned int nrChannels = 0;
  unsigned int nrStations = 0;
  unsigned int nrSamples = 0;
  isa::OpenCL::KernelConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    vectorSize = args.getSwitchArgument< unsigned int >("-vector");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    nrChannels = args.getSwitchArgument< unsigned int >("-channels");
    nrStations = args.getSwitchArgument< unsigned int >("-stations");
    nrSamples = args.getSwitchArgument< unsigned int >("-samples");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -padding ... -iterations ... -vector ... -max_threads ... -max_items ... -channels ... -stations ... -samples ..." << std::endl;
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
  unsigned int nrBaselines = (nrStations * (nrStations + 1)) / 2;
  std::vector< inputDataType > input(nrChannels * nrStations * nrSamples * nrPolarizations * 2);
  std::vector< inputDataType > output(nrChannels * nrBaselines * nrPolarizations * nrPolarizations * 2), output_c(nrChannels * nrBaselines * nrPolarizations * nrPolarizations * 2);
  std::vector< unsigned int > baselineMap(nrBaselines * 2);
  cl::Buffer input_d, output_d, baselineMap_d;

  // Populate data structures
  srand(time(0));
  for ( unsigned int channel = 0; channel < nrChannels; channel++ ) {
    for ( unsigned int station = 0; station < nrStations; station++ ) {
      for ( unsigned int sample = 0; sample < nrSamples; sample++ ) {
        for ( unsigned int polarization = 0; polarization < nrPolarizations; polarization++ ) {
          input[(channel * nrStations * nrSamples * nrPolarizations * 2) + (station * nrSamples * nrPolarizations * 2) + (sample * nrPolarizations * 2) + (polarization * 2)] = rand() % magicValue;
          input[(channel * nrStations * nrSamples * nrPolarizations * 2) + (station * nrSamples * nrPolarizations * 2) + (sample * nrPolarizations * 2) + (polarization * 2) + 1] = rand() % magicValue;
        }
      }
    }
  }
  // Compute CPU control results
  std::fill(output.begin(), output.end(), 0);
  TuneBench::correlator(input, output_c, nrChannels, nrStations, nrSamples, nrPolarizations);
  TuneBench::generateBaselineMap(baselineMap, nrStations);

  std::cout << std::fixed << std::endl;
  std::cout << "# nrChannels nrStations nrSamples nrPolarizations nrThreadsD0 nrItemsD0 nrItemsD1 GFLOP/s time stdDeviation COV" << std::endl << std::endl;

  for ( unsigned int threads = vectorSize; threads <= maxThreads; threads += vectorSize ) {
    conf.setNrThreadsD0(threads);
    for ( unsigned int threads = 1; (conf.getNrThreadsD0() * threads) <= maxThreads; threads++ ) {
      conf.setNrThreadsD2(threads);
      if ( nrChannels % conf.getNrThreadsD2() != 0 ) {
        continue;
      }
      for ( unsigned int items = 1; items <= maxItems; items++ ) {
        conf.setNrItemsD0(items);
        if ( nrSamples % (conf.getNrThreadsD0() * conf.getNrItemsD0()) != 0 ) {
          continue;
        }
        for ( unsigned int items = 1; items * 18 <= maxItems; items++ ) {
          conf.setNrItemsD1(items);
          if ( nrStations % conf.getNrItemsD1() != 0 ) {
            continue;
          }

          // Generate kernel
          double gflops = isa::utils::giga(static_cast< uint64_t >(nrChannels) * nrSamples * nrBaselines * 32);
          cl::Event clEvent;
          cl::Kernel * kernel;
          isa::utils::Timer timer;
          std::string * code = TuneBench::getCorrelatorOpenCL(conf, inputDataName, padding, nrChannels, nrStations, nrSamples, nrPolarizations);

          if ( reInit ) {
            delete clQueues;
            clQueues = new std::vector< std::vector< cl::CommandQueue > >();
            isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
            try {
              initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &input, &input_d, nrChannels * nrBaselines * nrPolarizations * nrPolarizations * 2, &output_d, &baselineMap, &baselineMap_d);
            } catch ( cl::Error & err ) {
              return -1;
            }
            reInit = false;
          }
          try {
            kernel = isa::OpenCL::compile("correlator", *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
          } catch ( isa::OpenCL::OpenCLError & err ) {
            std::cerr << err.what() << std::endl;
            delete code;
            break;
          }
          delete code;

          cl::NDRange global(nrSamples / conf.getNrItemsD0(), nrBaselines / conf.getNrItemsD1(), nrChannels);
          cl::NDRange local(conf.getNrThreadsD0(), 1, conf.getNrThreadsD2());

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
            clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(inputDataType), reinterpret_cast< void * >(output.data()), 0, &clEvent);
            clEvent.wait();
          } catch ( cl::Error & err ) {
            reInit = true;
            std::cerr << "OpenCL kernel execution error (";
            std::cerr << conf.print();
            std::cerr << "): ";
            std::cerr << std::to_strin(err.err()) << std::endl;
            delete kernel;
            if ( err.err() == -4 || err.err() == -61 ) {
              return -1;
            }
            break;
          }
          delete kernel;

          bool error = false;
          for ( unsigned int item = 0; item < output_c.size(); item++ ) {
            if ( !isa::utils::same(output[item], output_c[item]) ) {
              std::cerr << "Output error (" << conf.print() << ")." << std::endl;
              error = true;
              break;
            }
          }
          if ( error ) {
            continue;
          }

          std::cout << nrChannels << " " << nrStations << " " << nrSamples << " " << nrPolarizations << " " ;
          std::cout << conf.print() << " ";
          std::cout << std::setprecision(3);
          std::cout << gflops / timer.getAverageTime() << " ";
          std::cout << std::setprecision(6);
          std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
        }
      }
    }
  }
  std::cout << std::endl;

  return 0;
}

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * input, cl::Buffer * input_d, const unsigned int outputSize, cl::Buffer * output_d, std::vector< unsigned int > * baselineMap, cl::Buffer * baselineMap_d) {
  try {
    *input_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, input->size() * sizeof(inputDataType), 0, 0);
    *output_d = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, outputSize * sizeof(inputDataType), 0, 0);
    *baselineMap_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, baselineMap->size() * sizeof(unsigned int), 0, 0);
    clQueue->enqueueWriteBuffer(*input_d, CL_FALSE, 0, input->size() * sizeof(inputDataType), reinterpret_cast< void * >(input->data()));
    clQueue->enqueueWriteBuffer(*baselineMap+d, CL_FALSE, 0, baselineMap->size() * sizeof(unsigned int), reinterpret_cast< void * >(baselineMap->data()));
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error (memory initialization): " << std::to_string(err.err()) << "." << std::endl;
    throw;
  }
}

