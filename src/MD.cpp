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

#include <MD.hpp>
#include <utils.hpp>

namespace TuneBench {

std::string * getMDOpenCL(const isa::OpenCL::KernelConf & conf, const std::string & dataName, const unsigned int nrAtoms, const float LJ1, const float LJ2) {
  std::string * code = new std::string();
  std::string LJ1_s = isa::utils::toString(LJ1);
  std::string LJ2_s = isa::utils::toString(LJ2);

  if ( LJ1_s.find(".") == std::string::npos ) {
    LJ1_s += ".0f";
  } else {
    LJ1_s += "f";
  }
  if ( LJ2_s.find(".") == std::string::npos ) {
    LJ2_s += ".0f";
  } else {
    LJ2_s += "f";
  }
  // Begin kernel's code
  *code = "__kernel void MD(__global const " + dataName + "4 * const restrict input, __global " + dataName + "4 * const restrict output) {\n"
    "<%DEFPOSITION%>"
    "<%DEFNEIGHBOR%>"
    "<%DEF%>"
    "for ( unsigned int neighbor = 0; neighbor < " + isa::utils::toString(nrAtoms) + "; neighbor += " + isa::utils::toString(conf.getNrItemsD1()) + " ) {\n"
    + dataName + " inverseDistance = 0.0;\n"
    + dataName + " force = 0.0;\n"
    "<%LOADNEIGHBOR%>"
    "<%COMPUTE%>"
    "}\n"
    "// Optional in-thread reduction\n"
    "<%REDUCE%>"
    "// Store\n"
    "<%STORE%>"
    "}\n";
  std::string defPosition_sTemplate = dataName + "4 position<%NUMD0%> = {0.0f, 0.0f, 0.0f, 0.0f};\n";
  std::string defNeighbor_sTemplate = dataName + "4 neighbor<%NUMD1%> = {0.0f, 0.0f, 0.0f, 0.0f};\n";
  std::string def_sTemplate = dataName + "4 accumulator<%NUMD0%>x<%NUMD1%> = {0.0f, 0.0f, 0.0f, 0.0f};\n";
  std::string loadNeighbor_sTemplate = "neighbor<%NUMD1%> = input[neighbor + <%OFFSETD1%>];\n";
  std::string compute_sTemplate = "inverseDistance = 1.0f / (((position<%NUMD0%>.x - neighbor<%NUMD1%>.x) * (position<%NUMD0%>.x - neighbor<%NUMD1%>.x)) + ((position<%NUMD0%>.y - neighbor<%NUMD1%>.y) * (position<%NUMD0%>.y - neighbor<%NUMD1%>.y)) + ((position<%NUMD0%>.z - neighbor<%NUMD1%>.z) * (position<%NUMD0%>.z - neighbor<%NUMD1%>.z)));\n"
    "force = (inverseDistance * inverseDistance * inverseDistance * inverseDistance) * ((" + LJ1_s + " * (inverseDistance * inverseDistance * inverseDistance)) - " + LJ2_s + ");\n"
    "accumulator<%NUMD0%>x<%NUMD1%>.x += (position<%NUMD0%>.x - neighbor<%NUMD1%>.x) * force;\n"
    "accumulator<%NUMD0%>x<%NUMD1%>.y += (position<%NUMD0%>.y - neighbor<%NUMD1%>.y) * force;\n"
    "accumulator<%NUMD0%>x<%NUMD1%>.z += (position<%NUMD0%>.z - neighbor<%NUMD1%>.z) * force;\n";
  std::string reduce_sTemplate = "accumulator<%NUMD0%>x0 += accumulator<%NUMD0%>x<%NUMD1%>;\n";
  std::string store_sTemplate = "output[get_global_id(0) + <%OFFSETD0%>] = accumulator<%NUMD0%>x0;\n";
  // End kernel's code

  std::string * defPosition_s = new std::string();
  std::string * defNeighbor_s = new std::string();
  std::string * def_s = new std::string();
  std::string * loadNeighbor_s = new std::string();
  std::string * compute_s = new std::string();
  std::string * reduce_s = new std::string();
  std::string * store_s = new std::string();
  std::string empty_s("");

  for ( unsigned int d0 = 0; d0 < conf.getNrItemsD0(); d0++ ) {
    std::string d0_s = isa::utils::toString(d0);
    std::string offsetD0_s = isa::utils::toString(d0 * conf.getNrThreadsD0());
    std::string * temp = 0;

    temp = isa::utils::replace(&defPosition_sTemplate, "<%NUMD0%>", d0_s);
    defPosition_s->append(*temp);
    delete temp;
    for ( unsigned int d1 = 0; d1 < conf.getNrItemsD1(); d1++ ) {
      std::string d1_s = isa::utils::toString(d1);
      std::string offsetD1_s = isa::utils::toString(d1);

      temp = isa::utils::replace(&def_sTemplate, "<%NUMD1%>", d1_s);
      def_s->append(*temp);
      delete temp;
      temp = isa::utils::replace(&compute_sTemplate, "<%NUMD1%>", d1_s);
      if ( d1 == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD1%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD1%>", offsetD1_s, true);
      }
      compute_s->append(*temp);
      delete temp;
      if ( d1 > 0 ) {
        temp = isa::utils::replace(&reduce_sTemplate, "<%NUMD1%>", d1_s);
        reduce_s->append(*temp);
        delete temp;
      }
    }
    def_s = isa::utils::replace(def_s, "<%NUMD0%>", d0_s, true);
    compute_s = isa::utils::replace(compute_s, "<%NUMD0%>", d0_s, true);
    if ( d0 == 0 ) {
      compute_s = isa::utils::replace(compute_s, " + <%OFFSETD0%>", empty_s, true);
    } else {
      compute_s = isa::utils::replace(compute_s, "<%OFFSETD0%>", offsetD0_s, true);
    }
    if ( conf.getNrItemsD1() > 1 ) {
      reduce_s = isa::utils::replace(reduce_s, "<%NUMD0%>", d0_s, true);
    }
    temp = isa::utils::replace(&store_sTemplate, "<%NUMD0%>", d0_s);
    if ( d0 == 0 ) {
      temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetD0_s, true);
    }
    store_s->append(*temp);
    delete temp;
  }
  for ( unsigned int d1 = 0; d1 < conf.getNrItemsD1(); d1++ ) {
    std::string d1_s = isa::utils::toString(d1);
    std::string offsetD1_s = isa::utils::toString(d1);
    std::string * temp = 0;

    temp = isa::utils::replace(&defNeighbor_sTemplate, "<%NUMD1%>", d1_s);
    defNeighbor_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&loadNeighbor_sTemplate, "<%NUMD1%>", d1_s);
    if ( d1 == 0 ) {
      temp = isa::utils::replace(temp, " + <%OFFSETD1%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSETD1%>", offsetD1_s, true);
    }
    loadNeighbor_s->append(*temp);
    delete temp;
  }

  code = isa::utils::replace(code, "<%DEFPOSITION%>", *defPosition_s, true);
  code = isa::utils::replace(code, "<%DEFNEIGHBOR%>", *defNeighbor_s, true);
  code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
  code = isa::utils::replace(code, "<%LOADNEIGHBOR%>", *loadNeighbor_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  code = isa::utils::replace(code, "<%REDUCE%>", *reduce_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete defPosition_s;
  delete defNeighbor_s;
  delete def_s;
  delete loadNeighbor_s;
  delete compute_s;
  delete reduce_s;
  delete store_s;

  return code;
}

} // TuneBench

