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
#include <Reduction.hpp>


int main(int argc, char * argv[]) {
  TuneBench::ReductionConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);

    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threads_d0"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-items_d0"));
    conf.setNrItemsPerBlock(args.getSwitchArgument< unsigned int >("-items_per_block"));
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -threads_d0 ... -items_d0 ... -items_per_block ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::string * code = TuneBench::getReductionOpenCL(conf, inputDataName, outputDataName);
  std::cout << std::endl;
  std::cout << *code << std::endl;
  std::cout << std::endl;

  return 0;
}

