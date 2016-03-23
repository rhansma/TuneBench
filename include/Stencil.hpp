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

#include <Kernel.hpp>


#ifndef STENCIL_HPP
#define STENCIL_HPP

namespace TuneBench {

class Stencil2DConf : public isa::OpenCL::KernelConf {
public:
  Stencil2DConf();
  // Get
  inline bool getLocalMemory() const;
  // Set
  inline void setLocalMemory(bool local);
  // utils
  std::string print() const;
private:
  bool useLocalMemory;
};

std::string * getStencil2DOpenCL(const Stencil2DConf & conf, const std::string & dataName, const unsigned int width, const unsigned int padding);


// Implementations
inline bool Stencil2DConf::getLocalMemory() const {
  return useLocalMemory;
}

inline void Stencil2DConf::setLocalMemory(bool local) {
  useLocalMemory = local;
}

}; // TuneBench

#endif // STENCIL_HPP

