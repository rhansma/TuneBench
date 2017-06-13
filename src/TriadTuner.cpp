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
#include <ctime>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <Triad.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>


namespace Triad {
    void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * A, cl::Buffer * A_d, std::vector< inputDataType > * B, cl::Buffer * B_d, cl::Buffer * C_d);

    std::string inputDataName("float");
    std::string outputDataName("float");

    int runKernel(int argc, char * argv[]) {
      bool reInit = true;
      unsigned int nrIterations = 0;
      unsigned int clPlatformID = 0;
      unsigned int clDeviceID = 0;
      unsigned int vectorSize = 0;
      unsigned int maxThreads = 0;
      unsigned int maxItems = 0;
      unsigned int maxVector = 0;
      unsigned int inputSize = 0;
      TuneBench::TriadConf conf;

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

      cl::Context clContext;
      std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
      std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
      std::vector< std::vector< cl::CommandQueue > > * clQueues = 0;

      // Allocate host memory
      std::vector< inputDataType > A(inputSize), B(inputSize), C(inputSize), C_control(inputSize);
      cl::Buffer A_d, B_d, C_d;

      srand(time(0));
      for ( unsigned int item = 0; item < A.size(); item++ ) {
        A[item] = rand() % factor;
        B[item] = rand() % factor;
      }
      std::fill(C.begin(), C.end(), factor);
      std::fill(C_control.begin(), C_control.end(), factor);
      TuneBench::triad(A, B, C_control, static_cast< inputDataType >(factor));

      std::cout << std::fixed << std::endl;
      std::cout << "# inputSize vector nrThreadsD0 nrItemsD0 GB/s time stdDeviation COV" << std::endl << std::endl;

      for ( unsigned int threads = vectorSize; threads <= maxThreads; threads += vectorSize ) {
        conf.setNrThreadsD0(threads);
        for ( unsigned int items = 1; items <= maxItems; items++ ) {
          conf.setNrItemsD0(items);
          for ( unsigned int vector = 1; vector <= maxVector; vector++ ) {
            conf.setVector(vector);
            if ( inputSize % (conf.getNrThreadsD0() * conf.getNrItemsD0() * conf.getVector()) != 0 ) {
              continue;
            } else if ( conf.getNrItemsD0() * conf.getVector() > maxItems ) {
              break;
            }

            // Generate kernel
            double gbytes = isa::utils::giga(static_cast< uint64_t >(inputSize) * sizeof(inputDataType) * 3.0);
            cl::Event clEvent;
            cl::Kernel * kernel;
            isa::utils::Timer timer;
            std::string * code = TuneBench::getTriadOpenCL(conf, inputDataName, factor);

            if ( reInit ) {
              delete clQueues;
              clQueues = new std::vector< std::vector< cl::CommandQueue > >();
              isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
              try {
                initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &A, &A_d, &B, &B_d, &C_d);
              } catch ( cl::Error & err ) {
                return -1;
              }
              reInit = false;
            }
            try {
              kernel = isa::OpenCL::compile("triad", *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
            } catch ( isa::OpenCL::OpenCLError & err ) {
              std::cerr << err.what() << std::endl;
              delete code;
              break;
            }
            delete code;

            cl::NDRange global(inputSize / (conf.getNrItemsD0() * conf.getVector()));
            cl::NDRange local(conf.getNrThreadsD0());

            kernel->setArg(0, A_d);
            kernel->setArg(1, B_d);
            kernel->setArg(2, C_d);

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
              clQueues->at(clDeviceID)[0].enqueueReadBuffer(C_d, CL_TRUE, 0, C.size() * sizeof(inputDataType), reinterpret_cast< void * >(C.data()), 0, &clEvent);
              clEvent.wait();
            } catch ( cl::Error & err ) {
              std::cerr << "OpenCL kernel execution error (";
              std::cerr << conf.print();
              std::cerr << "), (";
              std::cerr << isa::utils::toString(inputSize / conf.getNrItemsD0()) << "): ";
              std::cerr << isa::utils::toString(err.err()) << std::endl;
              delete kernel;
              if ( err.err() == -4 || err.err() == -61 ) {
                return -1;
              }
              reInit = true;
              break;
            }
            delete kernel;

            bool error = false;
            for ( unsigned int item = 0; item < C.size(); item++ ) {
              if ( !isa::utils::same(C[item], C_control[item]) ) {
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

    void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< inputDataType > * A, cl::Buffer * A_d, std::vector< inputDataType > * B, cl::Buffer * B_d, cl::Buffer * C_d) {
      try {
        *A_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, A->size() * sizeof(inputDataType), 0, 0);
        *B_d = cl::Buffer(clContext, CL_MEM_READ_ONLY, B->size() * sizeof(inputDataType), 0, 0);
        *C_d = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, A->size() * sizeof(inputDataType), 0, 0);
        clQueue->enqueueWriteBuffer(*A_d, CL_FALSE, 0, A->size() * sizeof(inputDataType), reinterpret_cast< void * >(A->data()));
        clQueue->enqueueWriteBuffer(*B_d, CL_FALSE, 0, B->size() * sizeof(inputDataType), reinterpret_cast< void * >(B->data()));
        clQueue->finish();
      } catch ( cl::Error & err ) {
        std::cerr << "OpenCL error (memory initialization): " << isa::utils::toString(err.err()) << "." << std::endl;
        throw;
      }
    }


}