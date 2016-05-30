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

#ifndef MD_HPP
#define MD_HPP

namespace TuneBench {

// Sequential
template< typename T > void MD(std::vector< T > & input, std::vector< T > & output, const unsigned int nrAtoms, const float LJ1, const float LJ2);
// OpenCL
std::string * getMDOpenCL(const isa::OpenCL::KernelConf & conf, const std::string & dataName, const unsigned int nrAtoms, const float LJ1, const float LJ2);


// Implementations
template< typename T > void MD(std::vector< T > & input, std::vector< T > & output, const unsigned int nrAtoms, const float LJ1, const float LJ2) {
  for ( unsigned int atom = 0; atom < nrAtoms; atom++ ) {
    T position[3];
    float accumulator[3] = {0.0f, 0.0f, 0.0f};

    position[0] = input[(atom * 4)];
    position[1] = input[(atom * 4) + 1];
    position[2] = input[(atom * 4) + 2];
    for ( unsigned int neighbor = 0; neighbor < nrAtoms; neighbor++ ) {
      T neighborPosition[3];
      float distance;
      float force;

      neighborPosition[0] = input[(neighbor * 4)];
      neighborPosition[1] = input[(neighbor * 4) + 1];
      neighborPosition[2] = input[(neighbor * 4) + 2];
      distance = 1.0f / (((position[0] - neighborPosition[0]) * (position[0] - neighborPosition[0])) + ((position[1] - neighborPosition[1]) * (position[1] - neighborPosition[1])) + ((position[2] - neighborPosition[2]) * (position[2] - neighborPosition[2])));
      force = (distance * distance * distance * distance) * ((LJ1 * (distance * distance * distance)) - LJ2);
      accumulator[0] += (position[0] - neighborPosition[0]) * force;
      accumulator[1] += (position[1] - neighborPosition[1]) * force;
      accumulator[2] += (position[2] - neighborPosition[2]) * force;
    }
    output[(atom * 4)] = accumulator[0];
    output[(atom * 4) + 1] = accumulator[1];
    output[(atom * 4) + 2] = accumulator[2];
    output[(atom * 4) + 3] = 0.0f;
  }
}

}; // TuneBench

#endif // MD_HPP

