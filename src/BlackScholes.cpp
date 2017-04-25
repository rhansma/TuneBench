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

#include <BlackScholes.hpp>

namespace TuneBench {

    BlackScholesConf::BlackScholesConf() : isa::OpenCL::KernelConf(), nrItemsPerBlock(1), vector(1) {}

    std::string BlackScholesConf::print() const {
      return isa::OpenCL::KernelConf::print();
      //return isa::utils::toString(nrItemsPerBlock) + ";" + isa::utils::toString(vector) + ";" + isa::OpenCL::KernelConf::print();
    }

    std::string * getBlackScholesOpenCL(const BlackScholesConf & conf, const std::string & inputDataName, const std::string & outputDataName) {
      std::string * code = new std::string();
      std::string vectorDataName;

      if ( conf.getVector() == 1 ) {
        vectorDataName = inputDataName;
      } else {
        vectorDataName = inputDataName + std::to_string(conf.getVector());
      }
      // Begin kernel's template
      std::ifstream t("src/BlackScholes.cl");
      std::stringstream buffer;
      buffer << t.rdbuf();

      code->assign(buffer.str());
      // End kernel's template

      return code;
    }

} // TuneBench

