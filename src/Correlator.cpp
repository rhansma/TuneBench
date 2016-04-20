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

#include <Correlator.hpp>

namespace TuneBench {

std::string * getCorrelatorOpenCL(const isa::OpenCL::KernelConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + " * const restrict output) {\n"
    "const unsigned int channel = get_group_id(2);\n"
    "<%DEFINE_STATIONS%>"
    "<%DEFINE_BASELINES%>"
    "__local " + dataName + " buffer[" + std::to_string(conf.getNrThreadsD0()) + "];\n"
    "\n"
    "// Compute\n"
    "for ( unsigned int sample = get_local_id(0); sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
    "<%LOAD_AND_COMPUTE%>"
    "}\n"
    "// Reduce and Store\n"
    "unsigned int threshold = 0;\n"
    "<%REDUCE_AND_STORE%>"
    "}\n";
  std::string defineStations_sTemplate = "const unsigned int station<%NUMD1%> = get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ";\n"
    + dataName + "4 sampleStation<%NUMD1%> = (0.0, 0.0, 0.0, 0.0);\n";
  std::string defineBaselines_sTemplate = dataName + "2 accumulator<%BASELINE%>00 = (0.0, 0.0);\n"
    + dataName + "2 accumulator<%BASELINE%>01 = (0.0, 0.0);\n"
    + dataName + "2 accumulator<%BASELINE%>10 = (0.0, 0.0);\n"
    + dataName + "2 accumulator<%BASELINE%>11 = (0.0, 0.0);\n";
  std::string load_sTemplate = "sampleStation<%NUMD1%> = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding) * nrPolarizations) + ") + (station * " + std::to_string(isa::utils::pad(nrSamples, padding) * nrPolarizations) + ") + ((sample + <%OFFSETD0%>) * " + std::to_string(nrPolarizations) + ")];\n";
  std::string compute_sTemplate = "accumulator<%BASELINE%>00.0 += (sampleStation<%NUMD1%>.0 * sampleStation<%STATION%>.0) - (sampleStation<%NUMD1%>.1 * (-sampleStation<%STATION%>.1));\n"
    "accumulator<%BASELINE%>00.1 += (sampleStation<%NUMD1%>.0 * (-sampleStation<%STATION%>.1)) + (sampleStation<%NUMD1%>.1 * sampleStation<%STATION%>.0);\n"
    "accumulator<%BASELINE%>01.0 += (sampleStation<%NUMD1%>.0 * sampleStation<%STATION%>.2) - (sampleStation<%NUMD1%>.1 * (-sampleStation<%STATION%>.3));\n"
    "accumulator<%BASELINE%>01.1 += (sampleStation<%NUMD1%>.0 * (-sampleStation<%STATION%>.3)) + (sampleStation<%NUMD1%>.1 * sampleStation<%STATION%>.2);\n"
    "accumulator<%BASELINE%>10.0 += (sampleStation<%NUMD1%>.2 * sampleStation<%STATION%>.0) - (sampleStation<%NUMD1%>.3 * (-sampleStation<%STATION%>.1));\n"
    "accumulator<%BASELINE%>10.1 += (sampleStation<%NUMD1%>.2 * (-sampleStation<%STATION%>.1)) + (sampleStation<%NUMD1%>.1 * sampleStation<%STATION%>.3);\n"
    "accumulator<%BASELINE%>11.0 += (sampleStation<%NUMD1%>.2 * sampleStation<%STATION%>.2) - (sampleStation<%NUMD1%>.3 * (-sampleStation<%STATION%>.3));\n"
    "accumulator<%BASELINE%>11.1 += (sampleStation<%NUMD1%>.2 * (-sampleStation<%STATION%>.3)) + (sampleStation<%NUMD1%>.3 * sampleStation<%STATION%>.2);\n";
  std::string reduceStore_sTemplate = "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[get_local_id(0)] = accumulator<%BASELINE%>00.0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>00.0 += buffer[item + threshold];\n"
    "buffer[item] = accumulator<%BASELINE%>00.0;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[((((station<%STATION%> * (station<%STATION%> + 1)) / 2) + station<%NUMD1%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ")] = accumulator<%BASELINE%>00.0;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[get_local_id(0)] = accumulator<%BASELINE%>00.1;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>00.1 += buffer[item + threshold];\n"
    "buffer[item] = accumulator<%BASELINE%>00.1;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[((((station<%STATION%> * (station<%STATION%> + 1)) / 2) + station<%NUMD1%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 1] = accumulator<%BASELINE%>00.1;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[get_local_id(0)] = accumulator<%BASELINE%>01.0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>01.0 += buffer[item + threshold];\n"
    "buffer[item] = accumulator<%BASELINE%>01.0;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[((((station<%STATION%> * (station<%STATION%> + 1)) / 2) + station<%NUMD1%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 2] = accumulator<%BASELINE%>01.0;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[get_local_id(0)] = accumulator<%BASELINE%>01.1;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>01.1 += buffer[item + threshold];\n"
    "buffer[item] = accumulator<%BASELINE%>01.1;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[((((station<%STATION%> * (station<%STATION%> + 1)) / 2) + station<%NUMD1%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 3] = accumulator<%BASELINE%>01.1;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[get_local_id(0)] = accumulator<%BASELINE%>10.0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>10.0 += buffer[item + threshold];\n"
    "buffer[item] = accumulator<%BASELINE%>10.0;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[((((station<%STATION%> * (station<%STATION%> + 1)) / 2) + station<%NUMD1%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 4] = accumulator<%BASELINE%>10.0;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[get_local_id(0)] = accumulator<%BASELINE%>10.1;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>10.1 += buffer[item + threshold];\n"
    "buffer[item] = accumulator<%BASELINE%>10.1;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[((((station<%STATION%> * (station<%STATION%> + 1)) / 2) + station<%NUMD1%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 5] = accumulator<%BASELINE%>10.1;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[get_local_id(0)] = accumulator<%BASELINE%>11.0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>11.0 += buffer[item + threshold];\n"
    "buffer[item] = accumulator<%BASELINE%>11.0;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[((((station<%STATION%> * (station<%STATION%> + 1)) / 2) + station<%NUMD1%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 6] = accumulator<%BASELINE%>11.0;\n"
    "}\n"
    "threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "buffer[get_local_id(0)] = accumulator<%BASELINE%>11.1;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "for ( unsigned int item = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( item < threshold ) {\n"
    "accumulator<%BASELINE%>11.1 += buffer[item + threshold];\n"
    "buffer[item] = accumulator<%BASELINE%>11.1;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[((((station<%STATION%> * (station<%STATION%> + 1)) / 2) + station<%NUMD1%>) * " + std::to_string(nrChannels * nrPolarizations * nrPolarizations * 2) + ") + (channel * " + std::to_string(nrPolarizations * nrPolarizations * 2) + ") + 7] = accumulator<%BASELINE%>11.1;\n"
    "}\n";
  // End kernel's template

  std::string * defineStations_s = new std::string();
  std::string * defineBaselines_s = new std::string();
  std::string * loadCompute_s = new std::string();
  std::string * reduceStore_s = new std::string();
  std::string * temp = 0;
  std::string empty_s = "";

  for ( unsigned int station0 = 0; station0 < conf.getNrItemsD1(); station0++ ) {
    std::string station0_s = std::to_string(station0);

    temp = isa::utils::replace(&defineStations_sTemplate, "<%NUMD1%>", station0_s);
    defineStations_s->append(*temp);
    delete temp;
    for ( unsigned int station1 = 0; station1 <= station0; station1++ ) {
      std::string station1_s = std::to_string(station1);
      std::string baseline_s = station0_s + "_" + station1_s;

      temp = isa::utils::replace(&defineBaselines_sTemplate, "<%BASELINE%>", baseline_s);
      defineBaselines_s->append(*temp);
      delete temp;
      temp = isa::utils::replace(&reduceStore_sTemplate, "<%BASELINE%>", baseline_s);
      temp = isa::utils::replace(temp, "<%NUMD1%>", station0_s, true);
      temp = isa::utils::replace(temp, "<%STATION%>", station1_s, true);
      reduceStore_s->append(*temp);
      delete temp;
    }
  }
  for ( unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++ ) {
    std::string offsetD0_s = std::to_string(sample * conf.getNrThreadsD0());

    for ( unsigned int station0 = 0; station0 < conf.getNrItemsD1(); station0++ ) {
      std::string station0_s = std::to_string(station0);

      temp = isa::utils::replace(&load_sTemplate, "<%NUMD1%>", station0_s);
      loadCompute_s->append(*temp);
      delete temp;
      for ( unsigned int station1 = 0; station1 <= station0; station1++ ) {
        std::string station1_s = std::to_string(station1);
        std::string baseline_s = station0_s + "_" + station1_s;

        temp = isa::utils::replace(&compute_sTemplate, "<%BASELINE%>", baseline_s);
        temp = isa::utils::replace(temp, "<%NUMD1%>", station0_s, true);
        temp = isa::utils::replace(temp, "<%STATION%>", station1_s, true);
        loadCompute_s->append(*temp);
        delete temp;
      }
    }
    if ( sample == 0 ) {
      loadCompute_s = isa::utils::replace(loadCompute_s, " + <%OFFSETD0%>", empty_s, true);
    } else {
      loadCompute_s = isa::utils::replace(loadCompute_s, "<%OFFSETD0%>", offsetD0_s, true);
    }
  }

  code = isa::utils::replace(code, "<%DEFINE_STATIONS%>", *defineStations_s, true);
  code = isa::utils::replace(code, "<%DEFINE_BASELINES%>", *defineBaselines_s, true);
  code = isa::utils::replace(code, "<%LOAD_AND_COMPUTE%>", *loadCompute_s, true);
  code = isa::utils::replace(code, "<%REDUCE_AND_STORE%>", *reduceStore_s, true);
  delete defineStations_s;
  delete defineBaselines_s;
  delete loadCompute_s;
  delete reduceStore_s;

  return code;
}

}; // TuneBench

