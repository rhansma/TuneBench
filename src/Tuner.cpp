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

#include <BlackScholes.hpp>
#include <Reduction.hpp>
#include <Correlator.hpp>
#include <MD.hpp>
#include <Stencil.hpp>
#include <Triad.hpp>

/**
 * Check result of the kernel, non-zero value means error occured
 * @param result    return code of kernel
 * @param kernel    name of the kernel
 */
void checkResult(int result, std::string kernel) {
  if(result == 0) {
    std::cout << "Successfully executed " << kernel << " Kernel" << std::endl;
  } else {
    std::cerr << "Error while executing " << kernel << " Kernel" << std::endl;
  }
}

int main(int argc, char * argv[]) {
  std::cout << "Starting kernels" << std::endl << std::endl;

  /* Execute kernels */
  std::cout << "BlackScholes" << std::endl;
  int BlackScholesResult = BlackScholes::runKernel(argc, argv);

  std::cout << std::endl << "Correlator" << std::endl;
  int CorrelatorResult = Correlator::runKernel(argc, argv);

  std::cout << std::endl << "MD" << std::endl;
  int MDResult = MD::runKernel(argc, argv);

  std::cout << std::endl << "Reduction" << std::endl;
  int ReductionResult = Reduction::runKernel(argc, argv);

  std::cout << std::endl << "Stencil" << std::endl;
  int StencilResult = Stencil::runKernel(argc, argv);

  std::cout << std::endl << "Triad" << std::endl;
  int TriadResult = Triad::runKernel(argc, argv);

  std::cout << std::endl;

  /* Check results */
  checkResult(BlackScholesResult, "BlackScholes");
  checkResult(CorrelatorResult, "Correlator");
  checkResult(MDResult, "MD");
  checkResult(ReductionResult, "Reduction");
  checkResult(StencilResult, "Stencil");
  checkResult(TriadResult, "Triad");

  return 0;
}