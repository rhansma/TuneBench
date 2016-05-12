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

#include <Reduction.hpp>

namespace TuneBench {

ReductionConf::ReductionConf() : isa::OpenCL::KernelConf(), nrItemsPerBlock(1) {}

std::string ReductionConf::print() const {
  return isa::utils::toString(nrItemsPerBlock) + " " + isa::utils::toString(vector) + " " + isa::OpenCL::KernelConf::print();
}

std::string * getReductionOpenCL(const ReductionConf & conf, const std::string & inputDataName, const std::string & outputDataName) {
  std::string * code = new std::string();
  std::string vectorDataName;

  if ( conf.getVector() == 1 ) {
    vectorDataName = inputDataName;
  } else {
    vectorDataName = inputDataName + std::to_string(conf.getVector());
  }
  // Begin kernel's template
  *code = "__kernel void reduction(__global const " + vectorDataName + " * const restrict input, __global " + outputDataName + " * const restrict output) {\n"
    "const unsigned int firstItem = (get_group_id(0) * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ") + get_local_id(0);\n"
    "__local " + outputDataName + " buffer[" + isa::utils::toString(conf.getNrThreadsD0()) + "];\n"
    "<%DEF%>"
    "\n"
    "// First compute phase\n"
    "for ( unsigned int item = firstItem; item < (firstItem + " + isa::utils::toString(conf.getNrItemsPerBlock()) + "); item += " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
    "<%COMPUTE%>"
    "}\n"
    "// In-thread reduce phase\n"
    + outputDataName + " accumulator = 0;\n"
    "<%REDUCE%>";
  if ( conf.getVector() == 1 ) {
    *code += "accumulator = accumulator0;\n";
  } else if ( conf.getVector() == 2 ) {
    *code += "accumulator = accumulator0.s0;\n"
      "accumulator += accumulator0.s1;\n";
  } else if ( conf.getVector() == 3 ) {
    *code += "accumulator = accumulator0.s0;\n"
      "accumulator += accumulator0.s1;\n"
      "accumulator += accumulator0.s2;\n";
  } else if ( conf.getVector() == 4 ) {
    *code += "accumulator = accumulator0.s0;\n"
      "accumulator += accumulator0.s1;\n"
      "accumulator += accumulator0.s2;\n"
      "accumulator += accumulator0.s3;\n";
  } else if ( conf.getVector() == 8 ) {
    *code += "accumulator = accumulator0.s0;\n"
      "accumulator += accumulator0.s1;\n"
      "accumulator += accumulator0.s2;\n"
      "accumulator += accumulator0.s3;\n"
      "accumulator += accumulator0.s4;\n"
      "accumulator += accumulator0.s5;\n"
      "accumulator += accumulator0.s6;\n"
      "accumulator += accumulator0.s7;\n";
  } else if ( conf.getVector() == 16 ) {
    *code += "accumulator = accumulator0.s0;\n"
      "accumulator += accumulator0.s1;\n"
      "accumulator += accumulator0.s2;\n"
      "accumulator += accumulator0.s3;\n"
      "accumulator += accumulator0.s4;\n"
      "accumulator += accumulator0.s5;\n"
      "accumulator += accumulator0.s6;\n"
      "accumulator += accumulator0.s7;\n"
      "accumulator += accumulator0.s8;\n"
      "accumulator += accumulator0.s9;\n"
      "accumulator += accumulator0.sA;\n"
      "accumulator += accumulator0.sB;\n"
      "accumulator += accumulator0.sC;\n"
      "accumulator += accumulator0.sD;\n"
      "accumulator += accumulator0.sE;\n"
      "accumulator += accumulator0.sF;\n";
  }
  *code += "buffer[get_local_id(0)] = accumulator;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Parallel reduce phase\n"
    "unsigned int threshold = " + isa::utils::toString(conf.getNrThreadsD0() / 2) + ";\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator += buffer[item + threshold];\n"
    "buffer[item] = accumulator;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "// Store\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[get_group_id(0)] = accumulator;\n"
    "}\n"
    "}\n";
  std::string def_sTemplate = vectorDataName + " accumulator<%NUM%> = 0;\n";
  std::string compute_sTemplate = "accumulator<%NUM%> += input[item + <%OFFSET%>];\n";
  std::string reduce_sTemplate = "accumulator0 += accumulator<%NUM%>;\n";
  // End kernel's template

  std::string * def_s = new std::string();
  std::string * compute_s = new std::string();
  std::string * reduce_s = new std::string();

  for ( unsigned int item = 0; item < conf.getNrItemsD0(); item++ ) {
    std::string item_s = isa::utils::toString(item);
    std::string offset_s = isa::utils::toString(item * conf.getNrThreadsD0());
    std::string * temp = 0;

    temp = isa::utils::replace(&def_sTemplate, "<%NUM%>", item_s);
    def_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&compute_sTemplate, "<%NUM%>", item_s);
    if ( item == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    compute_s->append(*temp);
    delete temp;
    if ( item > 0 ) {
      temp = isa::utils::replace(&reduce_sTemplate, "<%NUM%>", item_s);
      reduce_s->append(*temp);
      delete temp;
    }
  }
  code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  code = isa::utils::replace(code, "<%REDUCE%>", *reduce_s, true);
  delete def_s;
  delete compute_s;
  delete reduce_s;

  return code;
}

} // TuneBench

