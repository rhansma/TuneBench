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

#ifndef CONFIGURATION_TUNEBENCH_HPP
#define CONFIGURATION_TUNEBENCH_HPP

// Define the data types
typedef float inputDataType;
typedef float outputDataType;

// Magic value
const unsigned int magicValue = 42;

// Triad
const unsigned int factor = 42;

// MD
const float LJ1 = 1.5f;
const float LJ2 = 2.0f;

// Correlator
const unsigned int nrPolarizations = 2;

#endif // CONFIGURATION_TUNEBENCH_HPP

