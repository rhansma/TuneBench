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
        inline unsigned int getNrItemsPerBlock() const;
        inline unsigned int getVector() const;
        inline bool getLoopUnrolling() const;
        // Set
        inline void setNrItemsPerBlock(unsigned int items);
        inline void setVector(unsigned int vector);
        inline void setLoopUnrolling(bool unroll);
        // utils
        std::string print() const;
    private:
        unsigned int nrItemsPerBlock;
        unsigned int vector;
        bool loopUnrolling;
    };

    std::string * getBlackScholesOpenCL(const BlackScholesConf & conf, const std::string & inputDataName, const std::string & outputDataName);


// Implementations
    inline unsigned int BlackScholesConf::getNrItemsPerBlock() const {
      return nrItemsPerBlock;
    }

    inline unsigned int BlackScholesConf::getVector() const {
      return vector;
    }

    inline bool BlackScholesConf::getLoopUnrolling() const {
      return loopUnrolling;
    }

    inline void BlackScholesConf::setNrItemsPerBlock(unsigned int items) {
      nrItemsPerBlock = items;
    }

    inline void BlackScholesConf::setVector(unsigned int vector) {
      this->vector = vector;
    }

    inline void BlackScholesConf::setLoopUnrolling(bool unroll) {
      loopUnrolling = unroll;
    }

} // TuneBench

#endif

