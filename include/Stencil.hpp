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
#include <vector>

#include <utils.hpp>
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

// Sequential
template< typename T > void stencil2D(const std::vector< T > & input, std::vector< T > & output, const unsigned int width, const unsigned int padding);
// OpenCL
std::string * getStencil2DOpenCL(const Stencil2DConf & conf, const std::string & dataName, const unsigned int width, const unsigned int padding);


// Implementations
inline bool Stencil2DConf::getLocalMemory() const {
  return useLocalMemory;
}

inline void Stencil2DConf::setLocalMemory(bool local) {
  useLocalMemory = local;
}

template< typename T > void stencil2D(const std::vector< T > & input, std::vector< T > & output, const unsigned int width, const unsigned int padding) {
  for ( unsigned int y = 0; y < width; y++ ) {
    for ( unsigned int x = 0; x < width; x++ ) {
      output[(y * isa::utils::pad(width, padding)) + x] = (input[((y + 1) * isa::utils::pad(width + 2, padding)) + (x + 1)] * 0.25f) + (0.15f * (input[((y) * isa::utils::pad(width + 2, padding)) + (x + 1)] + input[((y + 1) * isa::utils::pad(width + 2, padding)) + (x)] + input[((y + 1) * isa::utils::pad(width + 2, padding)) + (x + 2)] + input[((y + 2) * isa::utils::pad(width + 2, padding)) + (x + 1)])) + (0.05f * (input[((y) * isa::utils::pad(width + 2, padding)) + (x)] + input[((y) * isa::utils::pad(width + 2, padding)) + (x + 2)] + input[((y + 2) * isa::utils::pad(width + 2, padding)) + (x)] + input[((y + 2) * isa::utils::pad(width + 2, padding)) + (x + 2)]));
    }
  }
}

}; // TuneBench

#endif // STENCIL_HPP

