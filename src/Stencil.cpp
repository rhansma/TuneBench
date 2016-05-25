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

#include <Stencil.hpp>
#include <utils.hpp>


namespace TuneBench {

Stencil2DConf::Stencil2DConf() : isa::OpenCL::KernelConf(), useLocalMemory(false) {}

std::string Stencil2DConf::print() const {
  return isa::utils::toString(useLocalMemory) + " " + isa::OpenCL::KernelConf::print();
}

std::string * getStencil2DOpenCL(const Stencil2DConf & conf, const std::string & dataName, const unsigned int width, const unsigned int padding) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void stencil2D(__global const " + dataName + " * const restrict input, __global " + dataName + " * const restrict output) {\n"
    "unsigned int outputRow = (get_group_id(1) * " + isa::utils::toString(conf.getNrThreadsD1() * conf.getNrItemsD1()) + ");\n"
    "unsigned int outputColumn = (get_group_id(0) * " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ");\n"
    "<%DEF%>";
  if ( conf.getLocalMemory() ) {
    *code += "__local " + dataName + " buffer[" + isa::utils::toString(((conf.getNrThreadsD1() * conf.getNrItemsD1()) + 2) * ((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2)) + "];\n"
      "\n"
      "for ( unsigned int localRow = get_local_id(1); localRow < " + std::to_string((conf.getNrThreadsD1() * conf.getNrItemsD1()) + 2) + "; localRow += " + std::to_string(conf.getNrThreadsD1()) + " ) {\n"
      "for ( unsigned int localColumn = get_local_id(0); localColumn < " + std::to_string((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + "; localColumn += " + std::to_string(conf.getNrThreadsD0()) + " ) {\n"
      "buffer[(localRow * " + std::to_string((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + localColumn] = input[((outputRow + localRow) * " + std::to_string(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + localColumn)];\n"
      "}\n"
      "}\n"
      "barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  *code += "// Compute items\n"
    "<%COMPUTE%>"
    "// Store\n"
    "<%STORE%>"
    "}\n";
  std::string def_sTemplate = dataName + " accumulator<%NUMD0%>x<%NUMD1%> = 0;\n";
  std::vector< std::string > compute_sTemplate(9);
  if ( conf.getLocalMemory() ) {
    compute_sTemplate[0] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.05f * buffer[((get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + <%OFFSETD0%>)];\n";
    compute_sTemplate[1] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.15f * buffer[((get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + 1 + <%OFFSETD0%>)];\n";
    compute_sTemplate[2] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.05f * buffer[((get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + 2 + <%OFFSETD0%>)];\n";
    compute_sTemplate[3] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.15f * buffer[((get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + <%OFFSETD0%>)];\n";
    compute_sTemplate[4] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.25f * buffer[((get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + 1 + <%OFFSETD0%>)];\n";
    compute_sTemplate[5] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.15f * buffer[((get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + 2 + <%OFFSETD0%>)];\n";
    compute_sTemplate[6] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.05f * buffer[((get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + <%OFFSETD0%>)];\n";
    compute_sTemplate[7] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.15f * buffer[((get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + 1 + <%OFFSETD0%>)];\n";
    compute_sTemplate[8] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.05f * buffer[((get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2) + ") + (get_local_id(0) + 2 + <%OFFSETD0%>)];\n";
  } else {
    compute_sTemplate[0] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.05f * input[((outputRow + get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + <%OFFSETD0%>)];\n";
    compute_sTemplate[1] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.15f * input[((outputRow + get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + 1 + <%OFFSETD0%>)];\n";
    compute_sTemplate[2] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.05f * input[((outputRow + get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + 2 + <%OFFSETD0%>)];\n";
    compute_sTemplate[3] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.15f * input[((outputRow + get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + <%OFFSETD0%>)];\n";
    compute_sTemplate[4] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.25f * input[((outputRow + get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + 1 + <%OFFSETD0%>)];\n";
    compute_sTemplate[5] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.15f * input[((outputRow + get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + 2 + <%OFFSETD0%>)];\n";
    compute_sTemplate[6] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.05f * input[((outputRow + get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + <%OFFSETD0%>)];\n";
    compute_sTemplate[7] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.15f * input[((outputRow + get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + 1 + <%OFFSETD0%>)];\n";
    compute_sTemplate[8] = "accumulator<%NUMD0%>x<%NUMD1%> += 0.05f * input[((outputRow + get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + get_local_id(0) + 2 + <%OFFSETD0%>)];\n";
  }
  std::string store_sTemplate = "output[((outputRow + get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width, padding)) + ") + (outputColumn + get_local_id(0) + <%OFFSETD0%>)] = accumulator<%NUMD0%>x<%NUMD1%>;\n";
  // End kernel's template

  std::string empty_s("");
  std::string * def_s = new std::string();
  std::string * compute_s = new std::string();
  std::string * store_s = new std::string();

  for ( unsigned int d1 = 0; d1 < conf.getNrItemsD1(); d1++ ) {
    std::string d1_s = isa::utils::toString(d1);
    std::string offsetd1_s = isa::utils::toString(d1 * conf.getNrThreadsD1());

    for ( unsigned int d0 = 0; d0 < conf.getNrItemsD0(); d0++ ) {
      std::string d0_s = isa::utils::toString(d0);
      std::string offsetd0_s = isa::utils::toString(d0 * conf.getNrThreadsD0());
      std::string * temp = 0;

      temp = isa::utils::replace(&def_sTemplate, "<%NUMD0%>", d0_s);
      if ( d0 == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
      }
      def_s->append(*temp);
      delete temp;
      temp = isa::utils::replace(&store_sTemplate, "<%NUMD0%>", d0_s);
      if ( d0 == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
      }
      store_s->append(*temp);
      delete temp;
    }

    def_s = isa::utils::replace(def_s, "<%NUMD1%>", d1_s, true);
    if ( d1 == 0 ) {
      def_s = isa::utils::replace(def_s, " + <%OFFSETD1%>", empty_s, true);
    } else {
      def_s = isa::utils::replace(def_s, "<%OFFSETD1%>", offsetd1_s, true);
    }
    compute_s = isa::utils::replace(compute_s, "<%NUMD1%>", d1_s, true);
    if ( d1 == 0 ) {
      compute_s = isa::utils::replace(compute_s, " + <%OFFSETD1%>", empty_s, true);
    } else {
      compute_s = isa::utils::replace(compute_s, "<%OFFSETD1%>", offsetd1_s, true);
    }
    store_s = isa::utils::replace(store_s, "<%NUMD1%>", d1_s, true);
    if ( d1 == 0 ) {
      store_s = isa::utils::replace(store_s, " + <%OFFSETD1%>", empty_s, true);
    } else {
      store_s = isa::utils::replace(store_s, "<%OFFSETD1%>", offsetd1_s, true);
    }
  }

  if ( conf.getLocalMemory() ) {
    for ( unsigned int d0 = 0; d0 < conf.getNrItemsD0(); d0++ ) {
      std::string d0_s = isa::utils::toString(d0);
      std::string offsetd0_s = isa::utils::toString(d0 * conf.getNrThreadsD0());
      std::string * temp = 0;

      for ( unsigned int computeStatement = 0; computeStatement < 9; computeStatement++ ) {
        for ( unsigned int d1 = 0; d1 < conf.getNrItemsD1(); d1++ ) {
          std::string d1_s = isa::utils::toString(d1);
          std::string offsetd1_s = isa::utils::toString(d1 * conf.getNrThreadsD1());

          temp = isa::utils::replace(&compute_sTemplate[computeStatement], "<%NUMD1%>", d1_s);
          if ( d1 == 0 ) {
            temp = isa::utils::replace(temp, " + <%OFFSETD1%>", empty_s, true);
          } else {
            temp = isa::utils::replace(temp, "<%OFFSETD1%>", offsetd1_s, true);
          }
          compute_s->append(*temp);
          delete temp;
        }
      }
      compute_s = isa::utils::replace(compute_s, "<%NUMD0%>", d0_s, true);
      if ( d0 == 0 ) {
        compute_s = isa::utils::replace(compute_s, " + <%OFFSETD0%>", empty_s, true);
      } else {
        compute_s = isa::utils::replace(compute_s, "<%OFFSETD0%>", offsetd0_s, true);
      }
    }
  } else {
    std::vector< std::vector< std::string > > computeStrings(2 + (conf.getNrThreadsD1() * (conf.getNrItemsD1() - 1)));

    for ( unsigned int item = 0; item < 2 + (conf.getNrThreadsD1() * (conf.getNrItemsD1() - 1)); item++ ) {
      computeStrings[item] = std::vector< std::string >(conf.getNrItemsD0());
    }
    for ( unsigned int computeStatement = 0; computeStatement < 9; computeStatement += 3 ) {

      for ( unsigned int d1 = 0; d1 < conf.getNrItemsD1(); d1++ ) {
        std::string d1_s = isa::utils::toString(d1);
        std::string offsetd1_s = isa::utils::toString(d1 * conf.getNrThreadsD1());

        for ( unsigned int d0 = 0; d0 < conf.getNrItemsD0(); d0++ ) {
          std::string d0_s = isa::utils::toString(d0);
          std::string offsetd0_s = isa::utils::toString(d0 * conf.getNrThreadsD0());
          std::string * temp = 0;

          temp = isa::utils::replace(&compute_sTemplate[computeStatement], "<%NUMD0%>", d0_s);
          if ( d0 == 0 ) {
            temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
          } else {
            temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
          }
          temp->append(*(isa::utils::replace(&compute_sTemplate[computeStatement + 1], "<%NUMD0%>", d0_s)));
          if ( d0 == 0 ) {
            temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
          } else {
            temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
          }
          temp->append(*(isa::utils::replace(&compute_sTemplate[computeStatement + 2], "<%NUMD0%>", d0_s)));
          if ( d0 == 0 ) {
            temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
          } else {
            temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
          }
          temp = isa::utils::replace(temp, "<%NUMD1%>", d1_s, true);
          if ( d1 == 0 ) {
            temp = isa::utils::replace(temp, " + <%OFFSETD1%>", empty_s, true);
          } else {
            temp = isa::utils::replace(temp, "<%OFFSETD1%>", offsetd1_s, true);
          }
          computeStrings.at((d1 * conf.getNrThreadsD1()) + (computeStatement / 3)).at(d0).append(*temp);
          delete temp;
        }
      }
    }
    for ( auto itemD1 = computeStrings.begin(); itemD1 != computeStrings.end(); ++itemD1 ) {
      for ( auto itemD0 = (*itemD1).begin(); itemD0 != (*itemD1).end(); ++itemD0 ) {
        compute_s->append(*itemD0);
      }
    }
  }

  code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete def_s;
  delete compute_s;
  delete store_s;

  return code;
}

} // TuneBench

