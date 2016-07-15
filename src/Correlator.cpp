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

CorrelatorConf::CorrelatorConf() : isa::OpenCL::KernelConf(), sequentialTime(true), parallelTime(false), constantMemory(false), width(1), height(1) {}

std::string CorrelatorConf::print() const {
  return std::to_string(sequentialTime) + " " + std::to_string(parallelTime) + " " + std::to_string(constantMemory) + " " + std::to_string(width) + " " + std::to_string(height) + " " + isa::OpenCL::KernelConf::print();
}

std::string * getCorrelatorOpenCLSequentialTime(const CorrelatorConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations, const unsigned int nrCells) {
  std::string * code = new std::string();

  // Begin kernel's template
  if ( conf.getConstantMemory() ) {
    *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + "8 * const restrict output, __constant const unsigned int * const restrict cellMapX, __constant const unsigned int * const restrict cellMapY) {\n";
  } else {
    *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + "8 * const restrict output, __global const unsigned int * const restrict cellMapX, __global const unsigned int * const restrict cellMapY) {\n";
  }
  *code += "const unsigned int cell = (get_group_id(0) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0);\n"
    "if ( cell < " + std::to_string(nrCells) + " ) {\n"
    "const unsigned int channel = (get_group_id(2) * " + std::to_string(conf.getNrThreadsD2()) + ") + get_local_id(2);\n"
    "const unsigned int baseStationX = cellMapX[cell];\n"
    "const unsigned int baseStationY = cellMapY[cell];\n"
    "<%DEFINE_STATION%>"
    "<%DEFINE_CELL%>"
    "\n"
    "// Compute\n"
    "for ( unsigned int sample = 0; sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrItemsD1()) + " ) {\n"
    "<%LOAD_AND_COMPUTE%>"
    "}\n"
    "<%STORE%>"
    "}\n"
    "}\n";
  std::string defineStationX_sTemplate = dataName + "4 sampleStationX<%STATION%> = (" + dataName + "4)(0.0);\n";
  std::string defineStationY_sTemplate = dataName + "4 sampleStationY<%STATION%> = (" + dataName + "4)(0.0);\n";
  std::string defineCell_sTemplate = dataName + "8 accumulator<%CELL%> = (" + dataName + "8)(0.0);\n";
  std::string loadX_sTemplate = "sampleStationX<%STATION%> = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + ((baseStationX + <%WIDTH%>) * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD1%>)];\n";
  std::string loadY_sTemplate = "sampleStationY<%STATION%> = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + ((baseStationY + <%HEIGHT%>) * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD1%>)];\n";
  std::vector< std::string > compute_sTemplate(4);
  compute_sTemplate[0] = "accumulator<%CELL%>.s0246 += sampleStationX<%STATIONX%>.s0022 * sampleStationY<%STATIONY%>.s0202;\n";
  compute_sTemplate[1] = "accumulator<%CELL%>.s0246 += sampleStationX<%STATIONX%>.s1133 * sampleStationY<%STATIONY%>.s1313;\n";
  compute_sTemplate[2] = "accumulator<%CELL%>.s1357 += sampleStationX<%STATIONX%>.s1133 * sampleStationY<%STATIONY%>.s0202;\n";
  compute_sTemplate[3] = "accumulator<%CELL%>.s1357 -= sampleStationX<%STATIONX%>.s0022 * sampleStationY<%STATIONY%>.s1313;\n";
  std::string store_sTemplate = "output[(((((baseStationY + <%HEIGHT%>) * (baseStationY + <%HEIGHT%> + 1)) / 2) + baseStationX + <%WIDTH%>) * " + std::to_string(nrChannels) + ") + channel] = accumulator<%CELL%>;\n";
  // End kernel's template

  std::string * defineStation_s = new std::string();
  std::string * defineCell_s = new std::string();
  std::string * loadCompute_s = new std::string();
  std::string * store_s = new std::string();
  std::string * temp = 0;
  std::string empty_s = "";

  for ( unsigned int width = 0; width < conf.getCellWidth(); width++ ) {
    std::string width_s = std::to_string(width);

    temp = isa::utils::replace(&defineStationX_sTemplate, "<%STATION%>", width_s);
    defineStation_s->append(*temp);
    delete temp;
  }
  for ( unsigned int height = 0; height < conf.getCellHeight(); height++ ) {
    std::string height_s = std::to_string(height);

    temp = isa::utils::replace(&defineStationY_sTemplate, "<%STATION%>", height_s);
    defineStation_s->append(*temp);
    delete temp;
  }
  for ( unsigned int width = 0; width < conf.getCellWidth(); width++ ) {
    for ( unsigned int height = 0; height < conf.getCellHeight(); height++ ) {
      std::string cell_s = std::to_string(width) + std::to_string(height);
      std::string width_s = std::to_string(width);
      std::string height_s = std::to_string(height);

      temp = isa::utils::replace(&defineCell_sTemplate, "<%CELL%>", cell_s);
      defineCell_s->append(*temp);
      delete temp;
      if ( width == 0 ) {
        temp = isa::utils::replace(&store_sTemplate, " + <%WIDTH%>", empty_s);
      } else {
        temp = isa::utils::replace(&store_sTemplate, "<%WIDTH%>", width_s);
      }
      if ( height == 0 ) {
        temp = isa::utils::replace(temp, " + <%HEIGHT%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%HEIGHT%>", height_s, true);
      }
      temp = isa::utils::replace(temp, "<%CELL%>", cell_s, true);
      store_s->append(*temp);
      delete temp;
    }
  }
  for ( unsigned int sample = 0; sample < conf.getNrItemsD1(); sample++ ) {
    std::string offsetD1_s = std::to_string(sample);

    for ( unsigned int width = 0; width < conf.getCellWidth(); width++ ) {
      std::string width_s = std::to_string(width);

      if ( width == 0 ) {
        temp = isa::utils::replace(&loadX_sTemplate, " + <%WIDTH%>", empty_s);
      } else {
        temp = isa::utils::replace(&loadX_sTemplate, "<%WIDTH%>", width_s);
      }
      temp = isa::utils::replace(temp, "<%STATION%>", width_s, true);
      loadCompute_s->append(*temp);
      delete temp;
    }
    for ( unsigned int height = 0; height < conf.getCellHeight(); height++ ) {
      std::string height_s = std::to_string(height);

      if ( height == 0 ) {
        temp = isa::utils::replace(&loadY_sTemplate, " + <%HEIGHT%>", empty_s);
      } else {
        temp = isa::utils::replace(&loadY_sTemplate, "<%HEIGHT%>", height_s);
      }
      temp = isa::utils::replace(temp, "<%STATION%>", height_s, true);
      loadCompute_s->append(*temp);
      delete temp;
    }
    if ( sample == 0 ) {
      loadCompute_s = isa::utils::replace(loadCompute_s, " + <%OFFSETD1%>", empty_s, true);
    } else {
      loadCompute_s = isa::utils::replace(loadCompute_s, "<%OFFSETD1%>", offsetD1_s, true);
    }
    for ( unsigned int computeStatement = 0; computeStatement < compute_sTemplate.size(); computeStatement++ ) {
      for ( unsigned int width = 0; width < conf.getCellWidth(); width++ ) {
        std::string width_s = std::to_string(width);

        for ( unsigned int height = 0; height < conf.getCellHeight(); height++ ) {
          std::string height_s = std::to_string(height);
          std::string cell_s = std::to_string(width) + std::to_string(height);

          temp = isa::utils::replace(&compute_sTemplate[computeStatement], "<%CELL%>", cell_s);
          temp = isa::utils::replace(temp, "<%STATIONX%>", width_s, true);
          temp = isa::utils::replace(temp, "<%STATIONY%>", height_s, true);
          loadCompute_s->append(*temp);
          delete temp;
        }
      }
    }
  }

  code = isa::utils::replace(code, "<%DEFINE_STATION%>", *defineStation_s, true);
  code = isa::utils::replace(code, "<%DEFINE_CELL%>", *defineCell_s, true);
  code = isa::utils::replace(code, "<%LOAD_AND_COMPUTE%>", *loadCompute_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete defineStation_s;
  delete defineCell_s;
  delete loadCompute_s;
  delete store_s;

  return code;
}

std::string * getCorrelatorOpenCLParallelTime(const CorrelatorConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations) {
  std::string * code = new std::string();

  // Begin kernel's template
  if ( conf.getConstantMemory() ) {
    *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + "8 * const restrict output, __constant  const unsigned int * const restrict cellMapX, __constant const unsigned int * const restrict cellMapY) {\n";
  } else {
    *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + "8 * const restrict output, __global const unsigned int * const restrict cellMapX, __global const unsigned int * const restrict cellMapY) {\n";
  }
  *code += "const unsigned int channel = (get_group_id(2) * " + std::to_string(conf.getNrThreadsD2()) + ") + get_local_id(2);\n"
    "const unsigned int baseStationX = cellMapX[get_group_id(1)];\n"
    "const unsigned int baseStationY = cellMapY[get_group_id(1)];\n"
    "<%DEFINE_STATION%>"
    "<%DEFINE_CELL%>"
    "__local " + dataName + "8 buffer[" + std::to_string(conf.getNrThreadsD0() * conf.getNrThreadsD2() * conf.getCellWidth() * conf.getCellHeight()) + "];\n"
    "\n"
    "// Compute\n"
    "for ( unsigned int sample = get_local_id(0); sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
    "<%LOAD_AND_COMPUTE%>"
    "}\n"
    "// Reduce\n"
    "unsigned int threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "<%REDUCE_LOAD%>"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "<%REDUCE_COMPUTE%>"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "<%STORE%>"
    "}\n"
    "}\n";
  std::string defineStationX_sTemplate = dataName + "4 sampleStationX<%STATION%> = (" + dataName + "4)(0.0);\n";
  std::string defineStationY_sTemplate = dataName + "4 sampleStationY<%STATION%> = (" + dataName + "4)(0.0);\n";
  std::string defineCell_sTemplate = dataName + "8 accumulator<%CELL%> = (" + dataName + "8)(0.0);\n";
  std::string loadX_sTemplate = "sampleStationX<%STATION%> = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + ((baseStationX + <%WIDTH%>) * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD0%>)];\n";
  std::string loadY_sTemplate = "sampleStationY<%STATION%> = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + ((baseStationY + <%HEIGHT%>) * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD0%>)];\n";
  std::vector< std::string > compute_sTemplate(4);
  compute_sTemplate[0] = "accumulator<%CELL%>.s0246 += sampleStationX<%STATIONX%>.s0022 * sampleStationY<%STATIONY%>.s0202;\n";
  compute_sTemplate[1] = "accumulator<%CELL%>.s0246 += sampleStationX<%STATIONX%>.s1133 * sampleStationY<%STATIONY%>.s1313;\n";
  compute_sTemplate[2] = "accumulator<%CELL%>.s1357 += sampleStationX<%STATIONX%>.s1133 * sampleStationY<%STATIONY%>.s0202;\n";
  compute_sTemplate[3] = "accumulator<%CELL%>.s1357 -= sampleStationX<%STATIONX%>.s0022 * sampleStationY<%STATIONY%>.s1313;\n";
  std::string reduceLoad_sTemplate = "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0() * conf.getCellWidth() * conf.getCellHeight()) + ") + get_local_id(0) + <%CELL_OFFSET%>] = accumulator<%CELL%>;\n";
  std::string reduceCompute_sTemplate = "accumulator<%CELL%> += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0() * conf.getCellWidth() * conf.getCellHeight()) + ") + item + threshold + <%CELL_OFFSET%>];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0() * conf.getCellWidth() * conf.getCellHeight()) + ") + item + <%CELL_OFFSET%>] = accumulator<%CELL%>;\n";
  std::string store_sTemplate = "output[(((((baseStationY + <%HEIGHT%>) * (baseStationY + <%HEIGHT%> + 1)) / 2) + baseStationX + <%WIDTH%>) * " + std::to_string(nrChannels) + ") + channel] = accumulator<%CELL%>;\n";
  // End kernel's template

  std::string * defineStation_s = new std::string();
  std::string * defineCell_s = new std::string();
  std::string * loadCompute_s = new std::string();
  std::string * reduceLoad_s = new std::string();
  std::string * reduceCompute_s = new std::string();
  std::string * store_s = new std::string();
  std::string * temp = 0;
  std::string empty_s = "";

  for ( unsigned int width = 0; width < conf.getCellWidth(); width++ ) {
    std::string width_s = std::to_string(width);

    temp = isa::utils::replace(&defineStationX_sTemplate, "<%STATION%>", width_s);
    defineStation_s->append(*temp);
    delete temp;
  }
  for ( unsigned int height = 0; height < conf.getCellHeight(); height++ ) {
    std::string height_s = std::to_string(height);

    temp = isa::utils::replace(&defineStationY_sTemplate, "<%STATION%>", height_s);
    defineStation_s->append(*temp);
    delete temp;
  }
  for ( unsigned int width = 0; width < conf.getCellWidth(); width++ ) {
    for ( unsigned int height = 0; height < conf.getCellHeight(); height++ ) {
      std::string cell_s = std::to_string(width) + std::to_string(height);
      std::string cellOffset_s = std::to_string(((width * conf.getCellHeight()) + height) * conf.getNrThreadsD0());
      std::string width_s = std::to_string(width);
      std::string height_s = std::to_string(height);

      temp = isa::utils::replace(&defineCell_sTemplate, "<%CELL%>", cell_s);
      defineCell_s->append(*temp);
      delete temp;
      if ( width == 0 ) {
        temp = isa::utils::replace(&store_sTemplate, " + <%WIDTH%>", empty_s);
      } else {
        temp = isa::utils::replace(&store_sTemplate, "<%WIDTH%>", width_s);
      }
      if ( height == 0 ) {
        temp = isa::utils::replace(temp, " + <%HEIGHT%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%HEIGHT%>", height_s, true);
      }
      temp = isa::utils::replace(temp, "<%CELL%>", cell_s, true);
      store_s->append(*temp);
      delete temp;
      temp = isa::utils::replace(&reduceLoad_sTemplate, "<%CELL%>", cell_s);
      if ( width + height == 0 ) {
        temp = isa::utils::replace(temp, " + <%CELL_OFFSET%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%CELL_OFFSET%>", cellOffset_s, true);
      }
      reduceLoad_s->append(*temp);
      delete temp;
      temp = isa::utils::replace(&reduceCompute_sTemplate, "<%CELL%>", cell_s);
      if ( width + height == 0 ) {
        temp = isa::utils::replace(temp, " + <%CELL_OFFSET%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%CELL_OFFSET%>", cellOffset_s, true);
      }
      reduceCompute_s->append(*temp);
      delete temp;
    }
  }
  for ( unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++ ) {
    std::string offsetD0_s = std::to_string(sample * conf.getNrThreadsD0());

    for ( unsigned int width = 0; width < conf.getCellWidth(); width++ ) {
      std::string width_s = std::to_string(width);

      if ( width == 0 ) {
        temp = isa::utils::replace(&loadX_sTemplate, " + <%WIDTH%>", empty_s);
      } else {
        temp = isa::utils::replace(&loadX_sTemplate, "<%WIDTH%>", width_s);
      }
      temp = isa::utils::replace(temp, "<%STATION%>", width_s, true);
      loadCompute_s->append(*temp);
      delete temp;
    }
    for ( unsigned int height = 0; height < conf.getCellHeight(); height++ ) {
      std::string height_s = std::to_string(height);

      if ( height == 0 ) {
        temp = isa::utils::replace(&loadY_sTemplate, " + <%HEIGHT%>", empty_s);
      } else {
        temp = isa::utils::replace(&loadY_sTemplate, "<%HEIGHT%>", height_s);
      }
      temp = isa::utils::replace(temp, "<%STATION%>", height_s, true);
      loadCompute_s->append(*temp);
      delete temp;
    }
    if ( sample == 0 ) {
      loadCompute_s = isa::utils::replace(loadCompute_s, " + <%OFFSETD0%>", empty_s, true);
    } else {
      loadCompute_s = isa::utils::replace(loadCompute_s, "<%OFFSETD0%>", offsetD0_s, true);
    }
    for ( unsigned int computeStatement = 0; computeStatement < compute_sTemplate.size(); computeStatement++ ) {
      for ( unsigned int width = 0; width < conf.getCellWidth(); width++ ) {
        std::string width_s = std::to_string(width);

        for ( unsigned int height = 0; height < conf.getCellHeight(); height++ ) {
          std::string height_s = std::to_string(height);
          std::string cell_s = std::to_string(width) + std::to_string(height);

          temp = isa::utils::replace(&compute_sTemplate[computeStatement], "<%CELL%>", cell_s);
          temp = isa::utils::replace(temp, "<%STATIONX%>", width_s, true);
          temp = isa::utils::replace(temp, "<%STATIONY%>", height_s, true);
          loadCompute_s->append(*temp);
          delete temp;
        }
      }
    }
  }

  code = isa::utils::replace(code, "<%DEFINE_STATION%>", *defineStation_s, true);
  code = isa::utils::replace(code, "<%DEFINE_CELL%>", *defineCell_s, true);
  code = isa::utils::replace(code, "<%LOAD_AND_COMPUTE%>", *loadCompute_s, true);
  code = isa::utils::replace(code, "<%REDUCE_LOAD%>", *reduceLoad_s, true);
  code = isa::utils::replace(code, "<%REDUCE_COMPUTE%>", *reduceCompute_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete defineStation_s;
  delete defineCell_s;
  delete loadCompute_s;
  delete reduceLoad_s;
  delete reduceCompute_s;
  delete store_s;

  return code;
}

std::string * getCorrelatorOpenCL(const CorrelatorConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations, const unsigned int nrCells) {
  if ( conf.getSequentialTime() ) {
    return getCorrelatorOpenCLSequentialTime(conf, dataName, padding, nrChannels, nrStations, nrSamples, nrPolarizations, nrCells);
  } else if ( conf.getParallelTime() ) {
    return getCorrelatorOpenCLParallelTime(conf, dataName, padding, nrChannels, nrStations, nrSamples, nrPolarizations);
  }
  return new std::string();
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

