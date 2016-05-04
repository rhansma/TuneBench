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

std::string * getCorrelatorOpenCL(const isa::OpenCL::KernelConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + " * const restrict output, __global const unsigned int * const restrict baselineMap) {\n"
    "const unsigned int channel = (get_group_id(2) * " + std::to_string(conf.getNrThreadsD2()) + ") + get_local_id(2);\n"
    "<%DEFINE%>"
    "__local " + dataName + " buffer[" + std::to_string(conf.getNrThreadsD0() * conf.getNrThreadsD2()) + "];\n"
    "\n"
    "// Compute\n"
    "for ( unsigned int sample = get_local_id(0); sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
    "<%LOAD_AND_COMPUTE%>"
    "}\n"
    "// Reduce and Store\n"
    "unsigned int threshold = 0;\n"
    "<%REDUCE_AND_STORE%>"
    "}\n";
  std::string define_sTemplate = "const unsigned int station<%STATION_X%> = baselineMap[(get_group_id(1) * " + std::to_string(conf.getNrItemsD1() * 2) + ") + <%BASELINE%>];\n"
    "const unsigned int station<%STATION_Y%> = baselineMap[(get_group_id(1) * " + std::to_string(conf.getNrItemsD1() * 2) + ") + <%BASELINE%> + 1];\n"
    + dataName + "4 sampleStation<%STATION_X%> = (0.0, 0.0, 0.0, 0.0);\n"
    + dataName + "4 sampleStation<%STATION_Y%> = (0.0, 0.0, 0.0, 0.0);\n"
    + dataName + "2 accumulator<%BASELINE%>00 = (0.0, 0.0);\n"
    + dataName + "2 accumulator<%BASELINE%>01 = (0.0, 0.0);\n"
    + dataName + "2 accumulator<%BASELINE%>10 = (0.0, 0.0);\n"
    + dataName + "2 accumulator<%BASELINE%>11 = (0.0, 0.0);\n";
  std::string load_sTemplate = "sampleStation<%STATION_X%> = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + (station<%STATION_X%> * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD0%>)];\n"
    "if ( station<%STATION_X%> == station<%STATION_Y%> ) {\n"
    "sampleStation<%STATION_Y%> = sampleStation<%STATION_X%>;\n"
    "} else {\n"
    "sampleStation<%STATION_Y%> = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + (station<%STATION_Y%> * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD0%>)];\n"
    "}\n";
  std::string compute_sTemplate = "accumulator<%BASELINE%>00.x += (sampleStation<%STATION_X%>.x * sampleStation<%STATION_Y%>.x) - (sampleStation<%STATION_X%>.y * (-sampleStation<%STATION_Y%>.y));\n"
    "accumulator<%BASELINE%>00.y += (sampleStation<%STATION_X%>.x * (-sampleStation<%STATION_Y%>.y)) + (sampleStation<%STATION_X%>.y * sampleStation<%STATION_Y%>.x);\n"
    "accumulator<%BASELINE%>01.x += (sampleStation<%STATION_X%>.x * sampleStation<%STATION_Y%>.z) - (sampleStation<%STATION_X%>.y * (-sampleStation<%STATION_Y%>.w));\n"
    "accumulator<%BASELINE%>01.y += (sampleStation<%STATION_X%>.x * (-sampleStation<%STATION_Y%>.w)) + (sampleStation<%STATION_X%>.y * sampleStation<%STATION_Y%>.z);\n"
    "accumulator<%BASELINE%>10.x += (sampleStation<%STATION_X%>.z * sampleStation<%STATION_Y%>.x) - (sampleStation<%STATION_X%>.w * (-sampleStation<%STATION_Y%>.y));\n"
    "accumulator<%BASELINE%>10.y += (sampleStation<%STATION_X%>.z * (-sampleStation<%STATION_Y%>.y)) + (sampleStation<%STATION_X%>.w * sampleStation<%STATION_Y%>.x);\n"
    "accumulator<%BASELINE%>11.x += (sampleStation<%STATION_X%>.z * sampleStation<%STATION_Y%>.z) - (sampleStation<%STATION_X%>.w * (-sampleStation<%STATION_Y%>.w));\n"
    "accumulator<%BASELINE%>11.y += (sampleStation<%STATION_X%>.z * (-sampleStation<%STATION_Y%>.w)) + (sampleStation<%STATION_X%>.w * sampleStation<%STATION_Y%>.z);\n";
  std::string reduceStore_sTemplate = "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0)] = accumulator<%BASELINE%>00.x;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>00.x += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item + threshold];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item] = accumulator<%BASELINE%>00.x;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ")] = accumulator<%BASELINE%>00.x;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0)] = accumulator<%BASELINE%>00.y;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>00.y += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item + threshold];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item] = accumulator<%BASELINE%>00.y;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 1] = accumulator<%BASELINE%>00.y;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0)] = accumulator<%BASELINE%>01.x;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>01.x += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item + threshold];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item] = accumulator<%BASELINE%>01.x;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 2] = accumulator<%BASELINE%>01.x;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0)] = accumulator<%BASELINE%>01.y;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>01.y += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item + threshold];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item] = accumulator<%BASELINE%>01.y;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 3] = accumulator<%BASELINE%>01.y;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0)] = accumulator<%BASELINE%>10.x;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>10.x += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item + threshold];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item] = accumulator<%BASELINE%>10.x;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 4] = accumulator<%BASELINE%>10.x;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0)] = accumulator<%BASELINE%>10.y;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>10.y += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item + threshold];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item] = accumulator<%BASELINE%>10.y;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 5] = accumulator<%BASELINE%>10.y;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0)] = accumulator<%BASELINE%>11.x;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>11.x += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item + threshold];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item] = accumulator<%BASELINE%>11.x;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 6] = accumulator<%BASELINE%>11.x;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + get_local_id(0)] = accumulator<%BASELINE%>11.y;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>11.y += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item + threshold];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0()) + ") + item] = accumulator<%BASELINE%>11.y;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 7] = accumulator<%BASELINE%>11.y;\n"
    "}\n";
  // End kernel's template

  std::string * define_s = new std::string();
  std::string * loadCompute_s = new std::string();
  std::string * reduceStore_s = new std::string();
  std::string * temp = 0;
  std::string empty_s = "";

  for ( unsigned int baseline = 0; baseline < conf.getNrItemsD1(); baseline++ ) {
    std::string baseline_s = std::to_string(baseline);
    std::string stationX_s = std::to_string(baseline * 2);
    std::string stationY_s = std::to_string((baseline * 2) + 1);

    if ( baseline == 0 ) {
      temp = isa::utils::replace(&define_sTemplate, " + <%BASELINE%>", empty_s);
      temp = isa::utils::replace(temp, "<%BASELINE%>", baseline_s, true);
    } else {
      temp = isa::utils::replace(&define_sTemplate, "<%BASELINE%>", baseline_s);
    }
    temp = isa::utils::replace(temp, "<%STATION_X%>", stationX_s, true);
    temp = isa::utils::replace(temp, "<%STATION_Y%>", stationY_s, true);
    define_s->append(*temp);
    delete temp;
    if ( baseline == 0 ) {
      temp = isa::utils::replace(&reduceStore_sTemplate, " + <%BASELINE%>", empty_s);
      temp = isa::utils::replace(temp, "<%BASELINE%>", baseline_s, true);
    } else {
      temp = isa::utils::replace(&reduceStore_sTemplate, "<%BASELINE%>", baseline_s);
    }
    reduceStore_s->append(*temp);
    delete temp;
  }
  for ( unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++ ) {
    std::string offsetD0_s = std::to_string(sample * conf.getNrThreadsD0());

    for ( unsigned int baseline = 0; baseline < conf.getNrItemsD1(); baseline++ ) {
      std::string stationX_s = std::to_string(baseline * 2);
      std::string stationY_s = std::to_string((baseline * 2) + 1);

      temp = isa::utils::replace(&load_sTemplate, "<%STATION_X%>", stationX_s);
      temp = isa::utils::replace(temp, "<%STATION_Y%>", stationY_s, true);
      if ( sample == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetD0_s, true);
      }
      loadCompute_s->append(*temp);
      delete temp;
    }
    for ( unsigned int baseline = 0; baseline < conf.getNrItemsD1(); baseline++ ) {
      std::string baseline_s = std::to_string(baseline);
      std::string stationX_s = std::to_string(baseline * 2);
      std::string stationY_s = std::to_string((baseline * 2) + 1);

      temp = isa::utils::replace(&compute_sTemplate, "<%BASELINE%>", baseline_s);
      temp = isa::utils::replace(temp, "<%STATION_X%>", stationX_s, true);
      temp = isa::utils::replace(temp, "<%STATION_Y%>", stationY_s, true);
      loadCompute_s->append(*temp);
      delete temp;
    }
  }

  code = isa::utils::replace(code, "<%DEFINE%>", *define_s, true);
  code = isa::utils::replace(code, "<%LOAD_AND_COMPUTE%>", *loadCompute_s, true);
  code = isa::utils::replace(code, "<%REDUCE_AND_STORE%>", *reduceStore_s, true);
  delete define_s;
  delete loadCompute_s;
  delete reduceStore_s;

  return code;
}

void generateBaselineMap(std::vector< unsigned int > & baselineMap, const unsigned int nrStations) {
  for ( unsigned int station0 = 0; station0 < nrStations; station0++ ) {
    for ( unsigned int station1 = 0; station1 <= station0; station1++ ) {
      baselineMap[(((station0 * (station0 + 1)) / 2) + station1) * 2] = station0;
      baselineMap[((((station0 * (station0 + 1)) / 2) + station1) * 2) + 1] = station1;
    }
  }
}

}; // TuneBench

