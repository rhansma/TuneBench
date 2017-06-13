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

#include <iostream>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <Stencil.hpp>


int main(int argc, char * argv[]) {
  unsigned int matrixWidth = 0;
  unsigned int padding = 0;
  TuneBench::Stencil2DConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);

    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threads_d0"));
    conf.setNrThreadsD1(args.getSwitchArgument< unsigned int >("-threads_d1"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-items_d0"));
    conf.setNrItemsD1(args.getSwitchArgument< unsigned int >("-items_d1"));
    conf.setLocalMemory(args.getSwitch("-local"));
    matrixWidth = args.getSwitchArgument< unsigned int >("-width");
    padding = args.getSwitchArgument< unsigned int >("-padding");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -threads_d0 ... -threads_d1 ... -items_d0 ... -items_d1 ... [-local] -width ... -padding ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::string inputDataName("float");
  std::string outputDataName("float");

  std::string * code = TuneBench::getStencil2DOpenCL(conf, inputDataName, matrixWidth, padding);
  std::cout << std::endl;
  std::cout << *code << std::endl;
  std::cout << std::endl;

  return 0;
}

