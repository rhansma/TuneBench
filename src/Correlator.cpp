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

CorrelatorConf::CorrelatorConf() : isa::OpenCL::KernelConf(), parallelTime(false), sequentialTime(false) {}

std::string CorrelatorConf::print() const {
  return std::to_string(parallelTime) + " " + std::to_string(sequentialTime) + " " + isa::OpenCL::KernelConf::print();
}

std::string * getCorrelatorParallelTimeOpenCL(const CorrelatorConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + "8 * const restrict output, __global const uint2 * const restrict baselineMap) {\n"
    "const unsigned int channel = (get_group_id(2) * " + std::to_string(conf.getNrThreadsD2()) + ") + get_local_id(2);\n"
    "<%DEFINE%>"
    "__local " + dataName + "8 buffer[" + std::to_string(conf.getNrThreadsD0() * conf.getNrThreadsD2() * conf.getNrItemsD1()) + "];\n"
    "\n"
    "// Compute\n"
    "for ( unsigned int sample = get_local_id(0); sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
    "<%LOAD_AND_COMPUTE%>"
    "}\n"
    "// Reduce and Store\n"
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
  std::string define_sTemplate = "const uint2 station<%STATION%> = baselineMap[(get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>];\n"
    + dataName + "4 sampleStation<%STATION%>X = (" + dataName + "4)(0.0, 0.0, 0.0, 0.0);\n"
    + dataName + "4 sampleStation<%STATION%>Y = (" + dataName + "4)(0.0, 0.0, 0.0, 0.0);\n"
    + dataName + "8 accumulator<%BASELINE%> = (" + dataName + "8)(0.0);\n";
  std::string load_sTemplate = "sampleStation<%STATION%>X = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + (station<%STATION%>.s0 * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD0%>)];\n"
    "sampleStation<%STATION%>Y = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + (station<%STATION%>.s1 * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD0%>)];\n";
  std::vector< std::string > compute_sTemplate(8);
  compute_sTemplate[0] = "accumulator<%BASELINE%>.s0 += (sampleStation<%STATION%>X.x * sampleStation<%STATION%>Y.x) - (sampleStation<%STATION%>X.y * (-sampleStation<%STATION%>Y.y));\n";
  compute_sTemplate[1] = "accumulator<%BASELINE%>.s1 += (sampleStation<%STATION%>X.x * (-sampleStation<%STATION%>Y.y)) + (sampleStation<%STATION%>X.y * sampleStation<%STATION%>Y.x);\n";
  compute_sTemplate[2] = "accumulator<%BASELINE%>.s2 += (sampleStation<%STATION%>X.x * sampleStation<%STATION%>Y.z) - (sampleStation<%STATION%>X.y * (-sampleStation<%STATION%>Y.w));\n";
  compute_sTemplate[3] = "accumulator<%BASELINE%>.s3 += (sampleStation<%STATION%>X.x * (-sampleStation<%STATION%>Y.w)) + (sampleStation<%STATION%>X.y * sampleStation<%STATION%>Y.z);\n";
  compute_sTemplate[4] = "accumulator<%BASELINE%>.s4 += (sampleStation<%STATION%>X.z * sampleStation<%STATION%>Y.x) - (sampleStation<%STATION%>X.w * (-sampleStation<%STATION%>Y.y));\n";
  compute_sTemplate[5] = "accumulator<%BASELINE%>.s5 += (sampleStation<%STATION%>X.z * (-sampleStation<%STATION%>Y.y)) + (sampleStation<%STATION%>X.w * sampleStation<%STATION%>Y.x);\n";
  compute_sTemplate[6] = "accumulator<%BASELINE%>.s6 += (sampleStation<%STATION%>X.z * sampleStation<%STATION%>Y.z) - (sampleStation<%STATION%>X.w * (-sampleStation<%STATION%>Y.w));\n";
  compute_sTemplate[7] = "accumulator<%BASELINE%>.s7 += (sampleStation<%STATION%>X.z * (-sampleStation<%STATION%>Y.w)) + (sampleStation<%STATION%>X.w * sampleStation<%STATION%>Y.z);\n";
  std::string reduceLoad_sTemplate = "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD1()) + ") + get_local_id(0) + <%BASELINE_OFFSET%>] = accumulator<%BASELINE%>;\n";
  std::string reduceCompute_sTemplate = "accumulator<%BASELINE%> += buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD1()) + ") + item + threshold + <%BASELINE_OFFSET%>];\n"
    "buffer[(get_local_id(2) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD1()) + ") + item + <%BASELINE_OFFSET%>] = accumulator<%BASELINE%>;\n";
  std::string store_sTemplate = "output[(((get_group_id(1) * " + std::to_string(conf.getNrItemsD1()) + ") + <%BASELINE%>) * " + std::to_string(nrChannels) + ") + channel] = accumulator<%BASELINE%>;\n";
  // End kernel's template

  std::string * define_s = new std::string();
  std::string * loadCompute_s = new std::string();
  std::string * reduceLoad_s = new std::string();
  std::string * reduceCompute_s = new std::string();
  std::string * store_s = new std::string();
  std::string * temp = 0;
  std::string empty_s = "";

  for ( unsigned int baseline = 0; baseline < conf.getNrItemsD1(); baseline++ ) {
    std::string baseline_s = std::to_string(baseline);
    std::string baselineOffset_s = std::to_string(baseline * conf.getNrThreadsD0());

    if ( baseline == 0 ) {
      temp = isa::utils::replace(&define_sTemplate, " + <%BASELINE%>", empty_s);
      temp = isa::utils::replace(temp, "<%BASELINE%>", baseline_s, true);
    } else {
      temp = isa::utils::replace(&define_sTemplate, "<%BASELINE%>", baseline_s);
    }
    temp = isa::utils::replace(temp, "<%STATION%>", baseline_s, true);
    define_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&reduceLoad_sTemplate, "<%BASELINE%>", baseline_s);
    if ( baseline == 0 ) {
      temp = isa::utils::replace(temp, " + <%BASELINE_OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%BASELINE_OFFSET%>", baselineOffset_s, true);
    }
    reduceLoad_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&reduceCompute_sTemplate, "<%BASELINE%>", baseline_s);
    if ( baseline == 0 ) {
      temp = isa::utils::replace(temp, " + <%BASELINE_OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%BASELINE_OFFSET%>", baselineOffset_s, true);
    }
    reduceCompute_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&store_sTemplate, "<%BASELINE%>", baseline_s);
    store_s->append(*temp);
    delete temp;
  }
  for ( unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++ ) {
    std::string offsetD0_s = std::to_string(sample * conf.getNrThreadsD0());

    for ( unsigned int baseline = 0; baseline < conf.getNrItemsD1(); baseline++ ) {
      std::string baseline_s = std::to_string(baseline);

      temp = isa::utils::replace(&load_sTemplate, "<%STATION%>", baseline_s);
      if ( sample == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetD0_s, true);
      }
      loadCompute_s->append(*temp);
      delete temp;
    }
    for ( unsigned int computeStatement = 0; computeStatement < 8; computeStatement++ ) {
      for ( unsigned int baseline = 0; baseline < conf.getNrItemsD1(); baseline++ ) {
        std::string baseline_s = std::to_string(baseline);

        temp = isa::utils::replace(&compute_sTemplate[computeStatement], "<%BASELINE%>", baseline_s);
        temp = isa::utils::replace(temp, "<%STATION%>", baseline_s, true);
        loadCompute_s->append(*temp);
        delete temp;
      }
    }
  }

  code = isa::utils::replace(code, "<%DEFINE%>", *define_s, true);
  code = isa::utils::replace(code, "<%LOAD_AND_COMPUTE%>", *loadCompute_s, true);
  code = isa::utils::replace(code, "<%REDUCE_LOAD%>", *reduceLoad_s, true);
  code = isa::utils::replace(code, "<%REDUCE_COMPUTE%>", *reduceCompute_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete define_s;
  delete loadCompute_s;
  delete reduceLoad_s;
  delete reduceCompute_s;
  delete store_s;

  return code;
}

std::string * getCorrelatorSequentialTimeOpenCL(const CorrelatorConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void correlator(__global const " + dataName + "4 * const restrict input, __global " + dataName + "8 * const restrict output, __global const uint2 * const restrict baselineMap) {\n"
    "const unsigned int baseline = (get_group_id(0) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n"
    "const unsigned int channel = (get_group_id(2) * " + std::to_string(conf.getNrThreadsD2()) + ") + get_local_id(2);\n"
    "<%DEFINE%>"
    "\n"
    "// Compute\n"
    "for ( unsigned int sample = 0; sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrItemsD1()) + " ) {\n"
    "<%LOAD_AND_COMPUTE%>"
    "}\n"
    "<%STORE%>"
    "}\n";
  std::string define_sTemplate = "const uint2 station<%STATION%> = baselineMap[(baseline + <%OFFSETD0%>)];\n"
    + dataName + "4 sampleStation<%STATION%>X = (" + dataName + "4)(0.0, 0.0, 0.0, 0.0);\n"
    + dataName + "4 sampleStation<%STATION%>Y = (" + dataName + "4)(0.0, 0.0, 0.0, 0.0);\n"
    + dataName + "8 accumulator<%BASELINE%> = (" + dataName + "8)(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);\n";
  std::string load_sTemplate = "sampleStation<%STATION%>X = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + (station<%STATION%>.s0 * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD1%>)];\n"
    "sampleStation<%STATION%>Y = input[(channel * " + std::to_string(nrStations * isa::utils::pad(nrSamples, padding / 4)) + ") + (station<%STATION%>.s1 * " + std::to_string(isa::utils::pad(nrSamples, padding / 4)) + ") + (sample + <%OFFSETD1%>)];\n";
  std::vector< std::string > compute_sTemplate(8);
  compute_sTemplate[0] = "accumulator<%BASELINE%>.s0 += (sampleStation<%STATION%>X.x * sampleStation<%STATION%>Y.x) - (sampleStation<%STATION%>X.y * (-sampleStation<%STATION%>Y.y));\n";
  compute_sTemplate[1] = "accumulator<%BASELINE%>.s1 += (sampleStation<%STATION%>X.x * (-sampleStation<%STATION%>Y.y)) + (sampleStation<%STATION%>X.y * sampleStation<%STATION%>Y.x);\n";
  compute_sTemplate[2] = "accumulator<%BASELINE%>.s2 += (sampleStation<%STATION%>X.x * sampleStation<%STATION%>Y.z) - (sampleStation<%STATION%>X.y * (-sampleStation<%STATION%>Y.w));\n";
  compute_sTemplate[3] = "accumulator<%BASELINE%>.s3 += (sampleStation<%STATION%>X.x * (-sampleStation<%STATION%>Y.w)) + (sampleStation<%STATION%>X.y * sampleStation<%STATION%>Y.z);\n";
  compute_sTemplate[4] = "accumulator<%BASELINE%>.s4 += (sampleStation<%STATION%>X.z * sampleStation<%STATION%>Y.x) - (sampleStation<%STATION%>X.w * (-sampleStation<%STATION%>Y.y));\n";
  compute_sTemplate[5] = "accumulator<%BASELINE%>.s5 += (sampleStation<%STATION%>X.z * (-sampleStation<%STATION%>Y.y)) + (sampleStation<%STATION%>X.w * sampleStation<%STATION%>Y.x);\n";
  compute_sTemplate[6] = "accumulator<%BASELINE%>.s6 += (sampleStation<%STATION%>X.z * sampleStation<%STATION%>Y.z) - (sampleStation<%STATION%>X.w * (-sampleStation<%STATION%>Y.w));\n";
  compute_sTemplate[7] = "accumulator<%BASELINE%>.s7 += (sampleStation<%STATION%>X.z * (-sampleStation<%STATION%>Y.w)) + (sampleStation<%STATION%>X.w * sampleStation<%STATION%>Y.z);\n";
  std::string store_sTemplate = "output[((baseline + <%OFFSETD0%>) * " + std::to_string(nrChannels) + ") + channel] = accumulator<%BASELINE%>;\n";
  // End kernel's template

  std::string * define_s = new std::string();
  std::string * loadCompute_s = new std::string();
  std::string * store_s = new std::string();
  std::string * temp = 0;
  std::string empty_s = "";

  for ( unsigned int baseline = 0; baseline < conf.getNrItemsD0(); baseline++ ) {
    std::string baseline_s = std::to_string(baseline);
    std::string offsetD0_s = std::to_string(baseline * conf.getNrThreadsD0());

    temp = isa::utils::replace(&define_sTemplate, "<%BASELINE%>", baseline_s);
    if ( baseline == 0 ) {
      temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetD0_s, true);
    }
    temp = isa::utils::replace(temp, "<%STATION%>", baseline_s, true);
    define_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&store_sTemplate, "<%BASELINE%>", baseline_s);
    if ( baseline == 0 ) {
      temp = isa::utils::replace(temp, " + <%OFFSETD0%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSETD0%>", offsetD0_s, true);
    }
    store_s->append(*temp);
    delete temp;
  }
  for ( unsigned int sample = 0; sample < conf.getNrItemsD1(); sample++ ) {
    std::string offsetD1_s = std::to_string(sample);

    for ( unsigned int baseline = 0; baseline < conf.getNrItemsD0(); baseline++ ) {
      std::string baseline_s = std::to_string(baseline);

      temp = isa::utils::replace(&load_sTemplate, "<%STATION%>", baseline_s);
      if ( sample == 0 ) {
        temp = isa::utils::replace(temp, " + <%OFFSETD1%>", empty_s, true);
      } else {
        temp = isa::utils::replace(temp, "<%OFFSETD1%>", offsetD1_s, true);
      }
      loadCompute_s->append(*temp);
      delete temp;
    }
    for ( unsigned int computeStatement = 0; computeStatement < 8; computeStatement++ ) {
      for ( unsigned int baseline = 0; baseline < conf.getNrItemsD0(); baseline++ ) {
        std::string baseline_s = std::to_string(baseline);

        temp = isa::utils::replace(&compute_sTemplate[computeStatement], "<%BASELINE%>", baseline_s);
        temp = isa::utils::replace(temp, "<%STATION%>", baseline_s, true);
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

std::string * getCorrelatorOpenCL(const CorrelatorConf & conf, const std::string & dataName, const unsigned int padding, const unsigned int nrChannels, const unsigned int nrStations, const unsigned int nrSamples, const unsigned int nrPolarizations) {
  if ( conf.getParallelTime() ) {
    return getCorrelatorParallelTimeOpenCL(conf, dataName, padding, nrChannels, nrStations, nrSamples, nrPolarizations);
  } else if ( conf.getSequentialTime() ) {
    return getCorrelatorSequentialTimeOpenCL(conf, dataName, padding, nrChannels, nrStations, nrSamples, nrPolarizations);
  } else {
    return new std::string();
  }
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

