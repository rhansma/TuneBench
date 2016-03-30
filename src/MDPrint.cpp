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
#include <MD.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>


int main(int argc, char * argv[]) {
  unsigned int nrAtoms = 0;
  float LJ1 = 1.5f;
  float LJ2 = 2.0f;
  isa::OpenCL::KernelConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);

    nrAtoms = args.getSwitchArgument< unsigned int >("-atoms");
    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threads_d0"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-items_d0"));
    conf.setNrItemsD1(args.getSwitchArgument< unsigned int >("-items_d1"));
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -threads_d0 ... -items_d0 ... -items_d1 ... -atoms ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::string * code = TuneBench::getMDOpenCL(conf, inputDataName, nrAtoms, LJ1, LJ2);
  std::cout << std::endl;
  std::cout << *code << std::endl;
  std::cout << std::endl;

  return 0;
}

