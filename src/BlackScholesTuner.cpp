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

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <BlackScholes.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

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

int main(int argc, char * argv[]) {
  // Application specific parameters
  const float                    R = 0.02f;
  const float                    V = 0.30f;

  unsigned int nrIterations = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  unsigned int vectorSize = 0;
  unsigned int maxThreads = 0;
  unsigned int maxItems = 0;
  unsigned int maxVector = 0;
  unsigned int inputSize = 0;
  bool loopUnrolling = false;
  TuneBench::BlackScholesConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);

    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    inputSize = args.getSwitchArgument< unsigned int >("-input_size");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    loopUnrolling = args.getSwitchArgument< bool >("-loop_unrolling");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -opencl_platform ... -opencl_device ... -iterations ... -input_size ... -max_threads ... --loop_unrolling ..." << std::endl;
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
  std::vector< inputDataType > call(inputSize), put(inputSize), S(inputSize), X(inputSize), T(inputSize);
  cl::Buffer call_d, put_d, S_d, X_d, T_d;

  /* Fill the input vectors */
  srand(2009);
  for(unsigned int i = 0; i < inputSize; i++){
    S[i]       = randFloat(5.0f, 30.0f);
    X[i]       = randFloat(1.0f, 100.0f);
    T[i]       = randFloat(0.25f, 10.0f);
  }

  std::cout << std::fixed << std::endl;
  std::cout << "inputSize outputSize *configuration* GB/s time stdDeviation COV" << std::endl << std::endl;

  conf.setLoopUnrolling(loopUnrolling);
  for(unsigned int threads = 2; threads <= maxThreads; threads *= 2) {
    conf.setNrThreadsD0(threads);

    // Generate kernel
    unsigned int outputSize = inputSize;
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


    cl::NDRange global(conf.getNrThreadsD0() * 480); // 60 * 1024
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
      //clQueues->at(clDeviceID)[0].enqueueReadBuffer(call_d, CL_TRUE, 0, output.size() * sizeof(outputDataType), reinterpret_cast< void * >(output.data()), 0, &clEvent);
      clEvent.wait();
    } catch ( cl::Error & err ) {
      std::cerr << "OpenCL kernel execution error (" << inputSize << ", " << outputSize << "), (";
      std::cerr << conf.print();
      std::cerr << "), (";
      std::cerr << isa::utils::toString(conf.getNrThreadsD0() * (inputSize / conf.getNrItemsPerBlock() / conf.getVector())) << "): ";
      std::cerr << isa::utils::toString(err.err()) << std::endl;
      delete kernel;
      if ( err.err() == -4 || err.err() == -61 ) {
        return -1;
      }
      return -2;
    }
    delete kernel;

    /*bool error = false;
    for ( auto item = output.begin(); item != output.end(); ++item ) {
      if ( !isa::utils::same(*item, (magicValue * (inputSize / outputSize))) ) {
        std::cerr << "Output error (" << inputSize << ", " << outputSize << ") (" << conf.print() << ")." << std::endl;
        error = true;
        break;
      }
    }
    if ( error ) {
      return -2;
    }*/

    std::cout << inputSize << " " << outputSize << " ";
    std::cout << conf.print() << " ";
    std::cout << std::setprecision(3);
    std::cout << gbytes / timer.getAverageTime() << " ";
    std::cout << std::setprecision(6);
    std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
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

