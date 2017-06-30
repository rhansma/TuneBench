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

#pragma once

namespace Triad {
    int runKernel(unsigned int clPlatformID, unsigned int clDeviceID, unsigned int nrIterations, unsigned int vectorSize, unsigned int maxThreads, unsigned int maxItems, unsigned int maxVector, unsigned int inputSize);
}

namespace TuneBench {

class TriadConf : public isa::OpenCL::KernelConf {
public:
  TriadConf();
  // Get
  unsigned int getVector() const;
  // Set
  void setVector(unsigned int vector);
  // utils
  std::string print() const;
private:
  unsigned int vector;
};

// Sequential
template< typename T > void triad(const std::vector< T > & A, const std::vector< T > & B, std::vector< T > & C, const T factor);
// OpenCL
template< typename T > std::string * getTriadOpenCL(TriadConf & conf, std::string & dataName, const T factor);


// Implementations
inline unsigned int TriadConf::getVector() const {
  return vector;
}

inline void TriadConf::setVector(unsigned int vector) {
  this->vector = vector;
}

template< typename T > void triad(const std::vector< T > & A, const std::vector< T > & B, std::vector< T > & C, const T factor) {
  for ( unsigned int item = 0; item < C.size(); item++ ) {
    C[item] = A[item] + (factor * B[item]);
  }
}

template< typename T > std::string * getTriadOpenCL(TriadConf & conf, std::string & dataName, const T factor) {
  std::string * code = new std::string();
  std::string empty_s = isa::utils::toString("");
  std::string factor_s = std::to_string(factor);
  std::string vectorDataName = dataName;

  if ( factor_s.find(".") == std::string::npos ) {
    factor_s += ".0";
  }
  if ( dataName == "float" ) {
    factor_s += "f";
  }
  if ( conf.getVector() > 1 ) {
    vectorDataName += std::to_string(conf.getVector());
  }
  // Begin kernel's template
  *code = "__kernel void triad(__global const " + vectorDataName + " * const restrict A, __global const " + vectorDataName + " * const restrict B, __global " + vectorDataName + " * const restrict C) {\n"
    "unsigned int item = (get_group_id(0) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n"
    "<%COMPUTE%>"
    "}\n";
  std::string compute_sTemplate = "C[item + <%OFFSET%>] = A[item + <%OFFSET%>] + (" + factor_s + " * B[item + <%OFFSET%>]);\n";
  // End kernel's template

  std::string * compute_s = new std::string();

  for ( unsigned int item = 0; item < conf.getNrItemsD0(); item++ ) {
    std::string offset_s = std::to_string(conf.getNrThreadsD0() * item);
    std::string * temp = 0;

    if ( item == 0 ) {
      temp = isa::utils::replace(&compute_sTemplate, " + <%OFFSET%>", empty_s);
    } else {
      temp = isa::utils::replace(&compute_sTemplate, "<%OFFSET%>", offset_s);
    }
    compute_s->append(*temp);
    delete temp;
  }

  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  delete compute_s;

  return code;
}

}; // TuneBench

