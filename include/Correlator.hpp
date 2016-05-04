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

const unsigned int nrPolarizations = 2;

namespace TuneBench {

// Sequential
template< typename T > void correlator(const std::vector< T > & input, std::vector< T > & output, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations);
// OpenCL
std::string * getCorrelatorOpenCL(const isa::OpenCL::KernelConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations);

void generateBaselineMap(std::vector< unsigned int > & baselineMap, const unsigned int nrStations);

// Implementations
template< typename T > void correlator(const std::vector< T > & input, std::vector< T > & output, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations) {
  for ( unsigned int channel = 0; channel < nrChannels; channel++ ) {
    for ( unsigned int station0 = 0; station0 < nrStations; station0++ ) {
      for ( unsigned int station1 = 0; station1 <= station0; station1++ ) {
        const unsigned int baseline = ((station1 * (station1 + 1)) / 2) + station0;

        for ( unsigned int polarization0 = 0; polarization0 < nrPolarizations; polarization0++ ) {
          for ( unsigned int polarization1 = 0; polarization1 < nrPolarizations; polarization1++ ) {
            T accumulator[2] = {0.0, 0.0};

            for ( unsigned int sample = 0; sample < nrSamples; sample++ ) {
              T item0[2] = {0.0, 0.0};
              T item1[2] = {0.0, 0.0};

              item0[0] = input[(channel * nrStations * isa::utils::pad(nrSamples * nrPolarizations * 2, padding)) + (station0 * isa::utils::pad(nrSamples * nrPolarizations * 2, padding)) + (sample * nrPolarizations * 2) + (polarization0 * 2)];
              item0[1] = input[(channel * nrStations * isa::utils::pad(nrSamples * nrPolarizations * 2, padding)) + (station0 * isa::utils::pad(nrSamples * nrPolarizations * 2, padding)) + (sample * nrPolarizations * 2) + (polarization0 * 2) + 1];
              item1[0] = input[(channel * nrStations * isa::utils::pad(nrSamples * nrPolarizations * 2, padding)) + (station1 * isa::utils::pad(nrSamples * nrPolarizations * 2, padding)) + (sample * nrPolarizations * 2) + (polarization1 * 2)];
              item1[1] = input[(channel * nrStations * isa::utils::pad(nrSamples * nrPolarizations * 2, padding)) + (station1 * isa::utils::pad(nrSamples * nrPolarizations * 2, padding)) + (sample * nrPolarizations * 2) + (polarization1 * 2) + 1];
              accumulator[0] += (item0[0] * item1[0]) - (item0[1] * (-item1[1]));
              accumulator[1] += ((item0[0] + item0[1]) * (item1[0] - item1[1])) - (item0[0] * item1[0]) - (item0[1] * (-item1[1]));
            }
            output[(baseline * nrChannels * nrPolarizations * nrPolarizations * 2) + (channel * nrPolarizations * nrPolarizations * 2) + (polarization0 * nrPolarizations * 2) + (polarization1 * 2)] = accumulator[0];
            output[(baseline * nrChannels * nrPolarizations * nrPolarizations * 2) + (channel * nrPolarizations * nrPolarizations * 2) + (polarization0 * nrPolarizations * 2) + (polarization1 * 2) + 1] = accumulator[1];
          }
        }
      }
    }
  }
}

}; // TuneBench

