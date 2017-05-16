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

#include <string>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <sstream>

#include <utils.hpp>
#include <Kernel.hpp>



#ifndef BLACKSCHOLES_HPP
#define BLACKSCHOLES_HPP

namespace TuneBench {

    class BlackScholesConf : public isa::OpenCL::KernelConf {
    public:
        BlackScholesConf();
        // Get
        inline unsigned int getVector() const;
        inline unsigned int getLoopUnrolling() const;
        inline unsigned int getInputSize() const;
        // Set
        inline void setVector(unsigned int vector);
        inline void setLoopUnrolling(unsigned int loopUnrolling);
        inline void setInputSize(unsigned int inputSize);
        // utils
        std::string print() const;
    private:
        unsigned int nrItemsPerBlock;
        unsigned int vector;
        unsigned int loopUnrolling;
        unsigned int inputSize;
    };

    std::string * getBlackScholesOpenCL(const BlackScholesConf & conf, const std::string & inputDataName, const std::string & outputDataName);


// Implementations

    inline unsigned int BlackScholesConf::getVector() const {
      return vector;
    }

    inline unsigned int BlackScholesConf::getLoopUnrolling() const {
      return loopUnrolling;
    }

    inline unsigned int BlackScholesConf::getInputSize() const {
      return inputSize;
    }

    inline void BlackScholesConf::setVector(unsigned int vector) {
      this->vector = vector;
    }

    inline void BlackScholesConf::setLoopUnrolling(unsigned int loopUnrolling) {
      if(loopUnrolling != 1 && loopUnrolling != 3 && loopUnrolling != 7 && loopUnrolling != 15) {
        loopUnrolling = 0;
      }

      this->loopUnrolling = loopUnrolling;
    }

    inline void BlackScholesConf::setInputSize(unsigned int inputSize) {
      this->inputSize = inputSize;
    }

} // TuneBench

#endif

