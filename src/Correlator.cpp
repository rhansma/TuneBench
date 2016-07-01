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

#include <cmath>

#include <Correlator.hpp>


namespace TuneBench {

CorrelatorConf::CorrelatorConf() : isa::OpenCL::KernelConf(), width(1), height(1) {}

std::string CorrelatorConf::print() const {
  return std::to_string(width) + " " + std::to_string(height) + " " + isa::OpenCL::KernelConf::print();
}

std::string * getCorrelatorOpenCL(const CorrelatorConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + "8 * const restrict output, __global const unsigned int * const restrict cellMapX, __global const unsigned int * const restrict cellMapY) {\n"
    "const unsigned int cell = (get_group_id(0) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n"
    "const unsigned int channel = (get_group_id(2) * " + std::to_string(conf.getNrThreadsD2()) + ") + get_local_id(2);\n"
    "const unsigned int stationX = cellMapX[cell];\n"
    "const unsigned int stationY = cellMapY[cell];\n"
    "<%DEFINE_STATION%>"
    "<%DEFINE_CELL%>"
    "\n"
    "// Compute\n"
    "for ( unsigned int sample = 0; sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrItemsD1()) + " ) {\n"
    "<%LOAD_AND_COMPUTE%>"
    "}\n"
    "<%STORE%>"
    "}\n";
  std::string defineStation_sTemplate = dataName + "4 sampleStation<%STATION%>X = (" + dataName + "4)(0.0);\n"
    + dataName + "4 sampleStation<%STATION%>Y = (" + dataName + "4)(0.0);\n";
  std::string defineCell_sTemplate = dataName + "8 accumulator<%CELL%> = (" + dataName + "8)(0.0);\n";
  std::string load_sTemplate = "sampleStation<%STATION%>X = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + ((stationX + <%WIDTH%>) * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD1%>)];\n"
    "sampleStation<%STATION%>Y = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + ((stationY + <%HEIGHT%>) * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD1%>)];\n";
  std::vector< std::string > compute_sTemplate(8);
  compute_sTemplate[0] = "accumulator<%CELL%>.s0 += (sampleStation<%STATION%>X.x * sampleStation<%STATION%>Y.x) - (sampleStation<%STATION%>X.y * (-sampleStation<%STATION%>Y.y));\n";
  compute_sTemplate[1] = "accumulator<%CELL%>.s1 += (sampleStation<%STATION%>X.x * (-sampleStation<%STATION%>Y.y)) + (sampleStation<%STATION%>X.y * sampleStation<%STATION%>Y.x);\n";
  compute_sTemplate[2] = "accumulator<%CELL%>.s2 += (sampleStation<%STATION%>X.x * sampleStation<%STATION%>Y.z) - (sampleStation<%STATION%>X.y * (-sampleStation<%STATION%>Y.w));\n";
  compute_sTemplate[3] = "accumulator<%CELL%>.s3 += (sampleStation<%STATION%>X.x * (-sampleStation<%STATION%>Y.w)) + (sampleStation<%STATION%>X.y * sampleStation<%STATION%>Y.z);\n";
  compute_sTemplate[4] = "accumulator<%CELL%>.s4 += (sampleStation<%STATION%>X.z * sampleStation<%STATION%>Y.x) - (sampleStation<%STATION%>X.w * (-sampleStation<%STATION%>Y.y));\n";
  compute_sTemplate[5] = "accumulator<%CELL%>.s5 += (sampleStation<%STATION%>X.z * (-sampleStation<%STATION%>Y.y)) + (sampleStation<%STATION%>X.w * sampleStation<%STATION%>Y.x);\n";
  compute_sTemplate[6] = "accumulator<%CELL%>.s6 += (sampleStation<%STATION%>X.z * sampleStation<%STATION%>Y.z) - (sampleStation<%STATION%>X.w * (-sampleStation<%STATION%>Y.w));\n";
  compute_sTemplate[7] = "accumulator<%CELL%>.s7 += (sampleStation<%STATION%>X.z * (-sampleStation<%STATION%>Y.w)) + (sampleStation<%STATION%>X.w * sampleStation<%STATION%>Y.z);\n";
  std::string store_sTemplate = "output[(((((stationY + <%HEIGHT%>) * (stationY + <%HEIGHT%> + 1)) / 2) + stationX + <%WIDTH%>) * " + std::to_string(nrChannels) + ") + channel] = accumulator<%CELL%>;\n";
  // End kernel's template

  std::string * defineStation_s = new std::string();
  std::string * defineCell_s = new std::string();
  std::string * loadCompute_s = new std::string();
  std::string * store_s = new std::string();
  std::string * temp = 0;
  std::string empty_s = "";

  for ( unsigned int station = 0; station < conf.getCellWidth() + conf.getCellHeight(); station++ ) {
  }
  for ( unsigned int cell = 0; cell < conf.getNrItemsD0(); cell++ ) {
    std::string cell_s = std::to_string(cell);
    std::string offsetD0_s = std::to_string(cell * conf.getNrThreadsD0());

    temp = isa::utils::replace(&define_sTemplate, "<%CELL%>", cell_s);
    if ( cell == 0 ) {
      temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetD0_s, true);
    }
    temp = isa::utils::replace(temp, "<%STATION%>", cell_s, true);
    define_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&store_sTemplate, "<%CELL%>", cell_s);
    if ( cell == 0 ) {
      temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetD0_s, true);
    }
    store_s->append(*temp);
    delete temp;
  }
  for ( unsigned int sample = 0; sample < conf.getNrItemsD1(); sample++ ) {
    std::string offsetD1_s = std::to_string(sample);

    for ( unsigned int cell = 0; cell < conf.getNrItemsD0(); cell++ ) {
      std::string cell_s = std::to_string(cell);

      temp = isa::utils::replace(&load_sTemplate, "<%STATION%>", cell_s);
      if ( sample == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD1%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD1%>", offsetD1_s, true);
      }
      loadCompute_s->append(*temp);
      delete temp;
    }
    for ( unsigned int computeStatement = 0; computeStatement < 8; computeStatement++ ) {
      for ( unsigned int cell = 0; cell < conf.getNrItemsD0(); cell++ ) {
        std::string cell_s = std::to_string(cell);

        temp = isa::utils::replace(&compute_sTemplate[computeStatement], "<%CELL%>", cell_s);
        temp = isa::utils::replace(temp, "<%STATION%>", cell_s, true);
        loadCompute_s->append(*temp);
        delete temp;
      }
    }
  }

  code = isa::utils::replace(code, "<%DEFINE%>", *define_s, true);
  code = isa::utils::replace(code, "<%LOAD_AND_COMPUTE%>", *loadCompute_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete define_s;
  delete loadCompute_s;
  delete store_s;

  return code;
}

unsigned int generateCellMap(const CorrelatorConf & conf, std::vector< unsigned int > & cellMapX, std::vector< unsigned int > & cellMapY, const unsigned int nrStations) {
  unsigned int nrCells = 0;

  cellMapX.clear();
  cellMapY.clear();
  for ( int stationY = nrStations - conf.getCellHeight(); stationY >= 0; stationY -= conf.getCellHeight() ) {
    for ( int stationX = 0; stationX + static_cast< int >(conf.getCellWidth()) - 1 <= stationY; stationX += conf.getCellWidth() ) {
      nrCells++;
      cellMapX.push_back(stationX);
      cellMapY.push_back(stationY);
    }
  }

  return nrCells;
}

}; // TuneBench

