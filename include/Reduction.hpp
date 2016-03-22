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

#include <string>

#include <utils.hpp>
#include <Kernel.hpp>


#ifndef REDUCTION_HPP
#define REDUCTION_HPP

namespace TuneBench {

class ReductionConf : public isa::OpenCL::KernelConf {
public:
  ReductionConf();
  // Get
  inline unsigned int getNrItemsPerBlock() const;
  // Set
  inline void setNrItemsPerBlock(unsigned int items);
  // utils
  std::string print() const;
private:
  unsigned int nrItemsPerBlock;
};

std::string * getReductionOpenCL(const ReductionConf & conf, const std::string & inputDataName, const std::string & outputDataName);


// Implementations
inline unsigned int ReductionConf::getNrItemsPerBlock() const {
  return nrItemsPerBlock;
}

inline void ReductionConf::setNrItemsPerBlock(unsigned int items) {
  nrItemsPerBlock = items;
}

} // TuneBench

#endif

