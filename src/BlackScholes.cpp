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

#include <BlackScholes.hpp>

namespace TuneBench {

    BlackScholesConf::BlackScholesConf() : isa::OpenCL::KernelConf(), nrItemsPerBlock(1), vector(1) {}

    std::string BlackScholesConf::print() const {
      return isa::utils::toString(loopUnrolling)  + ";" + isa::OpenCL::KernelConf::print();
      //return isa::utils::toString(nrItemsPerBlock) + ";" + isa::utils::toString(vector) + ";" + isa::OpenCL::KernelConf::print();
    }

    std::string * getBlackScholesOpenCL(const BlackScholesConf & conf, const std::string & inputDataName, const std::string & outputDataName) {
      std::string * code = new std::string();
      std::string vectorDataName;

      if ( conf.getVector() == 1 ) {
        vectorDataName = inputDataName;
      } else {
        vectorDataName = inputDataName + std::to_string(conf.getVector());
      }
      // Begin kernel's template
      std::ifstream t("src/CL/BlackScholes.cl");
      std::stringstream buffer;
      buffer << t.rdbuf();

      code->assign(buffer.str());

      if(conf.getLoopUnrolling() >= 1) {
        std::string loop_sDecls = "for(unsigned int opt = (get_global_id(0) + (get_global_id(0) + <%OPT_SIZE%>)); opt < optN; opt += (get_global_size(0) + (get_global_size(0) * <%OPT_COUNT%>))) {\n"
            "float S; float X; float T; float sqrtT;"
            "float d1; float d2; float K; float CNDD1; "
            "float K2; float CNDD2; float expRT;"
            "<%IFD1%>";



        std::string * loop_sReplaced = 0;
        std::string * ifd1_s = new std::string();

        std::string opt_count_s = isa::utils::toString(conf.getLoopUnrolling());
        loop_sReplaced = isa::utils::replace(&loop_sDecls, "<%OPT_COUNT%>", opt_count_s);

        std::string opt_size_s = isa::utils::toString(conf.getLoopUnrolling() - 1);
        std::string t = loop_sReplaced->c_str();
        loop_sReplaced = isa::utils::replace(&t, "<%OPT_SIZE%>", opt_size_s);


        for(unsigned int i = 0; i < conf.getLoopUnrolling(); i++) {
          std::string * temp = 0;
          std::string opt_s = isa::utils::toString(i);
          std::string ifd1_sTemplate = "\n    S = d_S[opt + <%OPT%>];\n"
              "    X = d_X[opt + <%OPT%>];\n"
              "    T = d_T[opt + <%OPT%>];\n"
              "\n"
              "    sqrtT = SQRT(T);\n"
              "       d1 = (LOG(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);\n"
              "       d2 = d1 - V * sqrtT;\n"
              "        K = 1.0f / (1.0f + 0.2316419f * fabs(d1));\n"
              "\n"
              "    CNDD1 = RSQRT2PI * EXP(- 0.5f * d1 * d1) *\n"
              "                  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));\n"
              "\n"
              "    if(d1 > 0)\n"
              "      CNDD1 = 1.0f - CNDD1;\n"
              "\n"
              "    K2 = 1.0f / (1.0f + 0.2316419f * fabs(d2));\n"
              "\n"
              "    CNDD2 = RSQRT2PI * EXP(- 0.5f * d2 * d2) *\n"
              "                  (K2 * (A1 + K2 * (A2 + K2 * (A3 + K2 * (A4 + K2 * A5)))));\n"
              "\n"
              "    //Calculate Call and Put simultaneously\n"
              "    expRT = EXP(- R * T);\n"
              "    d_Call[opt + <%OPT%>] = (S * CNDD1 - X * expRT * CNDD2);\n"
              "    d_Put[opt + <%OPT%>] = (X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1));\n";


          temp = isa::utils::replace(&ifd1_sTemplate, "<%OPT%>", opt_s);
          ifd1_s->append(*temp);
          delete temp;
        }
        std::string * temp = 0;
        std::string * loop_s = new std::string();
        loop_s->append(*loop_sReplaced);
        std::string t1 = loop_s->c_str();
        loop_s = isa::utils::replace(&t1, "<%IFD1%>", *ifd1_s);

        loop_s->append("}");
        code = isa::utils::replace(code, "<%LOOP_UNROLL%>", *loop_s);
        delete loop_s;
        delete loop_sReplaced;
        delete temp;
      } else {
        std::string original_s("for(unsigned int opt = get_global_id(0); opt < optN; opt += get_global_size(0)) {\n"
                                   "    float S = d_S[opt];\n"
                                   "    float X = d_X[opt];\n"
                                   "    float T = d_T[opt];\n"
                                   "\n"
                                   "    float sqrtT = SQRT(T);\n"
                                   "    float    d1 = (LOG(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);\n"
                                   "    float    d2 = d1 - V * sqrtT;\n"
                                   "    float\n"
                                   "        K = 1.0f / (1.0f + 0.2316419f * fabs(d1));\n"
                                   "\n"
                                   "    float CNDD1 = RSQRT2PI * EXP(- 0.5f * d1 * d1) *\n"
                                   "                  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));\n"
                                   "\n"
                                   "    if(d1 > 0)\n"
                                   "      CNDD1 = 1.0f - CNDD1;\n"
                                   "\n"
                                   "    float K2 = 1.0f / (1.0f + 0.2316419f * fabs(d2));\n"
                                   "\n"
                                   "    float CNDD2 = RSQRT2PI * EXP(- 0.5f * d2 * d2) *\n"
                                   "                  (K2 * (A1 + K2 * (A2 + K2 * (A3 + K2 * (A4 + K2 * A5)))));\n"
                                   "\n"
                                   "    if(d2 > 0)\n"
                                   "      CNDD2 = 1.0f - CNDD2;\n"
                                   "\n"
                                   "    //Calculate Call and Put simultaneously\n"
                                   "    float expRT = EXP(- R * T);\n"
                                   "    d_Call[opt] = (S * CNDD1 - X * expRT * CNDD2);\n"
                                   "    d_Put[opt]  = (X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1));\n"
                                   "  }");
        code = isa::utils::replace(code, "<%LOOP_UNROLL%>", original_s);
      }

      // End kernel's template

      return code;
    }

} // TuneBench

