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
#include <Correlator.hpp>


int main(int argc, char * argv[]) {
  unsigned int nrChannels = 0;
  unsigned int nrStations = 0;
  unsigned int nrSamples = 0;
  unsigned int padding = 0;
  TuneBench::CorrelatorConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);

    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threads_d0"));
    conf.setNrThreadsD2(args.getSwitchArgument< unsigned int >("-threads_d2"));
    conf.setCell(args.getSwitchArgument< unsigned int >("-width"), args.getSwitchArgument< unsigned int >("-height"));
    conf.setNrItemsD0(conf.getCellWidth() * conf.getCellHeight());
    conf.setSequentialTime(args.getSwitch("-sequential_time"));
    conf.setParallelTime(args.getSwitch("-parallel_time"));
    if ( conf.getSequentialTime() ) {
      conf.setNrItemsD1(args.getSwitchArgument< unsigned int >("-items_d1"));
    } else if ( conf.getParallelTime() ) {
      conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-items_d0"));
    }
    nrChannels = args.getSwitchArgument< unsigned int >("-channels");
    nrStations = args.getSwitchArgument< unsigned int >("-stations");
    nrSamples = args.getSwitchArgument< unsigned int >("-samples");
    padding = args.getSwitchArgument< unsigned int >("-padding");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " [-sequential_time | -parallel_time] -threads_d0 ... -threads_d2 ... -width ... -height ... [-items_d1 ... | -items_d0 ...] -channels ... -stations ... -samples ... -padding ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  std::string * code = TuneBench::getCorrelatorOpenCL(conf, inputDataName, padding, nrChannels, nrStations, nrSamples, nrPolarizations);
  std::cout << std::endl;
  std::cout << *code << std::endl;
  std::cout << std::endl;

  return 0;
}


