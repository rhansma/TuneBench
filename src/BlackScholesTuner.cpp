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

#include <vector>
#include <iostream>
#include <exception>
#include <algorithm>
#include <iomanip>

#include <configuration.hpp>

#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <BlackScholes.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>


//using namespace TuneBench;

namespace BlackScholes {
//    using namespace TuneBench;
////////////////////////////////////////////////////////////////////////////////
// Random float helper
// NVIDIA SDK helper function
////////////////////////////////////////////////////////////////////////////////
float randFloat(float low, float high){
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}


void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * inputS, std::vector< inputDataType > * inputX, std::vector< inputDataType > * inputT,
                            cl::Buffer * S_d, cl::Buffer * X_d, cl::Buffer * T_d, cl::Buffer * call_d, cl::Buffer * put_d, const unsigned int outputSize);

    std::string inputDataName("float");
    std::string outputDataName("float");

int runKernel(unsigned int clPlatformID, unsigned int clDeviceID, unsigned int nrIterations, unsigned int inputSize,
              unsigned int maxThreads, unsigned int loopUnrolling) {
  // Application specific parameters
  const float                    R = 0.02f;
  const float                    V = 0.30f;

  TuneBench::BlackScholesConf conf;

  cl::Context clContext;
  std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
  std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
  std::vector< std::vector< cl::CommandQueue > > * clQueues = 0;

  // Allocate host memory
  std::vector< inputDataType > call(inputSize), put(inputSize), S(inputSize), X(inputSize), T(inputSize);
  cl::Buffer call_d, put_d, S_d, X_d, T_d;

  conf.setInputSize(inputSize);
  /* Fill the input vectors */
  srand(2009);
  for(unsigned int i = 0; i < inputSize; i++){
    S[i]       = randFloat(5.0f, 30.0f);
    X[i]       = randFloat(1.0f, 100.0f);
    T[i]       = randFloat(0.25f, 10.0f);
  }

  std::cout << std::fixed << std::endl;
  std::cout << "inputSize outputSize *configuration* GLOPS GB/s time stdDeviation COV" << std::endl << std::endl;

  for(unsigned int unroll = 0; unroll <= loopUnrolling; unroll++) {
    conf.setLoopUnrolling(unroll);
    /* If set value is not the same as retrieved value, the value is illegal thus no point in tuning */
    if(unroll != conf.getLoopUnrolling()) {
      continue;
    }
    std::cout << std::endl;
    for(unsigned int threads = 2; threads <= maxThreads; threads *= 2) {
      conf.setNrThreadsD0(threads);

      // Generate kernel
      unsigned int outputSize = inputSize;
      double gflops = isa::utils::giga(static_cast< uint64_t >(inputSize) * 56.0);
      double gbytes = isa::utils::giga((static_cast< uint64_t >(inputSize) * sizeof(inputDataType)) + (static_cast< uint64_t >(outputSize) * sizeof(outputDataType)));
      std::vector< outputDataType > output(outputSize);
      cl::Event clEvent;
      cl::Kernel * kernel;
      isa::utils::Timer timer;
      std::string * code = TuneBench::getBlackScholesOpenCL(conf, inputDataName, outputDataName);

      delete clQueues;
      clQueues = new std::vector< std::vector< cl::CommandQueue > >();
      isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
      try {
        initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &S, &X, &T, &S_d, &X_d, &T_d, &call_d, &put_d, outputSize);
      } catch ( cl::Error & err ) {
        return -1;
      }
      try {
        kernel = isa::OpenCL::compile("BlackScholes", *code, "-cl-fast-relaxed-math -Werror", clContext, clDevices->at(clDeviceID));
      } catch ( isa::OpenCL::OpenCLError & err ) {
        std::cerr << err.what() << std::endl;
        delete code;
        return -2;
      }
      delete code;

      std::vector<size_t> clMaxWorkItemSize = (clDevices->at(clDeviceID)).getInfo< CL_DEVICE_MAX_WORK_ITEM_SIZES >();
      unsigned int loopUnroll = std::max(conf.getLoopUnrolling() + 1, (unsigned int)1);
      int globalSize = conf.getNrThreadsD0() * (inputSize / conf.getNrThreadsD0() / loopUnroll); //640;
      int maxGlobalSize = (clMaxWorkItemSize[0] * clMaxWorkItemSize[1]);
      /* Stop when exceeding maximum work group size */
      if(conf.getNrThreadsD0() > clMaxWorkItemSize[0]) {
        std::cout << "Number of threads is greater than maximum possible value, stopping execution" << std::endl;
        break;
      }

      /* Limit global size to maximum global size supported by the hardware */
      if(globalSize > maxGlobalSize) {
        globalSize = maxGlobalSize;
      }

      cl::NDRange global(globalSize); // 60 * 1024
      cl::NDRange local(conf.getNrThreadsD0());

      kernel->setArg(0, call_d);
      kernel->setArg(1, put_d);
      kernel->setArg(2, S_d);
      kernel->setArg(3, X_d);
      kernel->setArg(4, T_d);
      kernel->setArg(5, R);
      kernel->setArg(6, V);
      kernel->setArg(7, inputSize);

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
        clEvent.wait();
      } catch ( cl::Error & err ) {
        std::cerr << "OpenCL kernel execution error (" << inputSize << ", " << outputSize << "), (";
        std::cerr << conf.print();
        std::cerr << "), (";
        std::cerr << isa::utils::toString(conf.getNrThreadsD0() * (inputSize / loopUnroll)) << "): ";
        std::cerr << isa::utils::toString(err.err()) << std::endl;
        delete kernel;
        if ( err.err() == -4 || err.err() == -61 ) {
          return -1;
        }
        return -2;
      }
      delete kernel;

      std::cout << inputSize << " " << outputSize << " ";
      std::cout << conf.print() << " ";
      std::cout << std::setprecision(3);
      std::cout << gflops / timer.getAverageTime() << " ";
      std::cout << gbytes / timer.getAverageTime() << " ";
      std::cout << std::setprecision(6);
      std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
    }
  }

  std::cout << std::endl;

  return 0;
}

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * inputS, std::vector< inputDataType > * inputX, std::vector< inputDataType > * inputT,
                            cl::Buffer * S_d, cl::Buffer * X_d, cl::Buffer * T_d, cl::Buffer * call_d, cl::Buffer * put_d, const unsigned int outputSize) {
  try {
    *S_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, inputS->size() * sizeof(inputDataType), 0, 0);
    *X_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, inputX->size() * sizeof(inputDataType), 0, 0);
    *T_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, inputT->size() * sizeof(inputDataType), 0, 0);
    *call_d = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, outputSize * sizeof(outputDataType), 0, 0);
    *put_d = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, outputSize * sizeof(outputDataType), 0, 0);

    clQueue->enqueueWriteBuffer(*S_d, CL_FALSE, 0, inputS->size() * sizeof(inputDataType), reinterpret_cast< void * >(inputS->data()));
    clQueue->enqueueWriteBuffer(*X_d, CL_FALSE, 0, inputX->size() * sizeof(inputDataType), reinterpret_cast< void * >(inputX->data()));
    clQueue->enqueueWriteBuffer(*T_d, CL_FALSE, 0, inputT->size() * sizeof(inputDataType), reinterpret_cast< void * >(inputT->data()));
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error (memory initialization): " << isa::utils::toString(err.err()) << "." << std::endl;
    throw;
  }
}
}
//}
/*
using namespace BlackScholes;
int main(int argc, char * argv[]) {
  return BlackScholes::runKernel(argc, argv);
}*/
