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

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <BlackScholes.hpp>


int main(int argc, char * argv[]) {
  TuneBench::BlackScholesConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);

    conf.setLoopUnrolling(args.getSwitchArgument< bool >("-loop_unrolling"));
    conf.setInputSize(args.getSwitchArgument< int >("-input_size"));
    conf.setNrThreadsD0(args.getSwitchArgument< bool >("-threads_d0"));
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " -loop_unrolling ... -input_size ... --threads_d0 ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::string * code = TuneBench::getBlackScholesOpenCL(conf, inputDataName, outputDataName);

  std::cout << std::endl;
  std::cout << *code << std::endl;
  std::cout << std::endl;

  return 0;
}

