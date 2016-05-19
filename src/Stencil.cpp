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
    "unsigned int outputRow = (get_group_id(1) * " + isa::utils::toString(conf.getNrThreadsD1() * conf.getNrItemsD1()) + ") + get_local_id(1);\n"
    "unsigned int outputColumn = (get_group_id(0) * " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n";
  if ( conf.getLocalMemory() ) {
    *code += "__local " + dataName + " buffer[" + isa::utils::toString(((conf.getNrThreadsD1() * conf.getNrItemsD1()) + 2) * isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + "];\n";
  }
  *code += "<%DEF%>";
  if ( conf.getLocalMemory() ) {
    *code += "// Load tile in local memory\n"
      "<%LOADMAIN%>"
      "if ( get_local_id(1) == 0 ) {\n"
      "<%LOADROWS%>"
      "}\n"
      "if ( get_local_id(0) < 2 ) {\n"
      "<%LOADCOLUMNS%>"
      "}\n"
      "if ( get_local_id(0) < 2 && get_local_id(1) == 0 ) {\n"
      "buffer[(" + isa::utils::toString(((conf.getNrThreadsD1() * conf.getNrItemsD1())) * isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ")] = input[ ((outputRow + " + isa::utils::toString(conf.getNrThreadsD1() * conf.getNrItemsD1()) + ") * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ")+ (outputColumn + " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ")];\n"
      "buffer[(" + isa::utils::toString(((conf.getNrThreadsD1() * conf.getNrItemsD1()) + 1) * isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ")] = input[ ((outputRow + " + isa::utils::toString((conf.getNrThreadsD1() * conf.getNrItemsD1()) + 1) + ") * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ")+ (outputColumn + " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ")];\n"
      "}\n"
      "barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  *code += "// Compute items\n"
    "<%COMPUTE%>"
    "// Store\n"
    "<%STORE%>"
    "}\n";
  std::string def_sTemplate = dataName + " accumulator<%NUMD0%>x<%NUMD1%> = 0;\n";
  std::string loadMain_sTemplate = "buffer[((get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + <%OFFSETD0%>)] = input[((outputRow + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + <%OFFSETD0%>)];\n";
  std::string loadRows_sTemplate = "buffer[(" + isa::utils::toString(((conf.getNrThreadsD1() * conf.getNrItemsD1())) * isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + <%OFFSETD0%>)] = input[((outputRow + " + isa::utils::toString(conf.getNrThreadsD1() * conf.getNrItemsD1()) + ") * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + <%OFFSETD0%>)];\n"
    "buffer[(" + isa::utils::toString(((conf.getNrThreadsD1() * conf.getNrItemsD1()) + 1) * isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + <%OFFSETD0%>)] = input[((outputRow + " + isa::utils::toString((conf.getNrThreadsD1() * conf.getNrItemsD1()) + 1) + ") * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + <%OFFSETD0%>)];\n";
  std::string loadColumns_sTemplate = "buffer[((get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ")] = input[((outputRow + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ")];\n";
  std::string compute_sTemplate;
  if ( conf.getLocalMemory() ) {
    compute_sTemplate = "accumulator<%NUMD0%>x<%NUMD1%> = (0.25f * buffer[((get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + 1 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.15f * buffer[((get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + 1 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.15f * buffer[((get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.15f * buffer[((get_local_id(1) + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + 2 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.15f * buffer[((get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + 1 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.05f * buffer[((get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.05f * buffer[((get_local_id(1) + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + 2 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.05f * buffer[((get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.05f * buffer[((get_local_id(1) + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(((conf.getNrThreadsD0() * conf.getNrItemsD0()) + 2), padding)) + ") + (get_local_id(0) + 2 + <%OFFSETD0%>)]);\n";
  } else {
    compute_sTemplate = "accumulator<%NUMD0%>x<%NUMD1%> = (0.25f * input[((outputRow + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + 1 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.15f * input[((outputRow + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + 1 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.15f * input[((outputRow + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.15f * input[((outputRow + 1 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + 2 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.15f * input[((outputRow + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + 1 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.05f * input[((outputRow + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.05f * input[((outputRow + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + 2 + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.05f * input[((outputRow + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + <%OFFSETD0%>)]);\n"
      "accumulator<%NUMD0%>x<%NUMD1%> += (0.05f * input[((outputRow + 2 + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width + 2, padding)) + ") + (outputColumn + 2 + <%OFFSETD0%>)]);\n";
  }
  std::string store_sTemplate = "output[((outputRow + <%OFFSETD1%>) * " + isa::utils::toString(isa::utils::pad(width, padding)) + ") + (outputColumn + <%OFFSETD0%>)] = accumulator<%NUMD0%>x<%NUMD1%>;\n";
  // End kernel's template

  std::string empty_s("");
  std::string * def_s = new std::string();
  std::string * loadMain_s = new std::string();
  std::string * loadRows_s = new std::string();
  std::string * loadColumns_s = new std::string();
  std::string * compute_s = new std::string();
  std::string * store_s = new std::string();

  for ( unsigned int d1 = 0; d1 < conf.getNrItemsD1(); d1++ ) {
    std::string d1_s = isa::utils::toString(d1);
    std::string offsetd1_s = isa::utils::toString(d1 * conf.getNrThreadsD1());
    std::string * temp = 0;

    for ( unsigned int d0 = 0; d0 < conf.getNrItemsD0(); d0++ ) {
      std::string d0_s = isa::utils::toString(d0);
      std::string offsetd0_s = isa::utils::toString(d0 * conf.getNrThreadsD0());

      temp = isa::utils::replace(&def_sTemplate, "<%NUMD0%>", d0_s);
      if ( d0 == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
      }
      def_s->append(*temp);
      delete temp;
      temp = isa::utils::replace(&loadMain_sTemplate, "<%NUMD0%>", d0_s);
      if ( d0 == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
      }
      loadMain_s->append(*temp);
      delete temp;
      temp = isa::utils::replace(&compute_sTemplate, "<%NUMD0%>", d0_s);
      if ( d0 == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
      }
      compute_s->append(*temp);
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
    loadMain_s = isa::utils::replace(loadMain_s, "<%NUMD1%>", d1_s, true);
    if ( d1 == 0 ) {
      loadMain_s = isa::utils::replace(loadMain_s, " + <%OFFSETD1%>", empty_s, true);
    } else {
      loadMain_s = isa::utils::replace(loadMain_s, "<%OFFSETD1%>", offsetd1_s, true);
    }
    temp = isa::utils::replace(&loadColumns_sTemplate, "<%NUMD1%>", d1_s);
    if ( d1 == 0 ) {
      temp = isa::utils::replace(temp, " + <%OFFSETD1%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSETD1%>", offsetd1_s, true);
    }
    loadColumns_s->append(*temp);
    delete temp;
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

  for ( unsigned int d0 = 0; d0 < conf.getNrItemsD0(); d0++ ) {
    std::string d0_s = isa::utils::toString(d0);
    std::string offsetd0_s = isa::utils::toString(d0 * conf.getNrThreadsD0());
    std::string * temp = 0;

    temp = isa::utils::replace(&loadRows_sTemplate, "<%NUMD0%>", d0_s);
    if ( d0 == 0 ) {
      temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetd0_s, true);
    }
    loadRows_s->append(*temp);
    delete temp;
  }

  code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
  code = isa::utils::replace(code, "<%LOADMAIN%>", *loadMain_s, true);
  code = isa::utils::replace(code, "<%LOADROWS%>", *loadRows_s, true);
  code = isa::utils::replace(code, "<%LOADCOLUMNS%>", *loadColumns_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete def_s;
  delete loadMain_s;
  delete loadRows_s;
  delete loadColumns_s;
  delete compute_s;
  delete store_s;

  return code;
}

} // TuneBench

