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

      if(conf.getLoopUnrolling() > 0) {
        std::string loop_sDecls = "for(unsigned int opt = get_global_id(0); (opt + <%OPT_COUNT%>) < optN; opt += get_global_size(0)) {\n"
            "float<%OPT_COUNT%> S; float<%OPT_COUNT%> X; float<%OPT_COUNT%> T; float<%OPT_COUNT%> sqrtT;"
            "float<%OPT_COUNT%> d1; float<%OPT_COUNT%> d2; float<%OPT_COUNT%> K; float<%OPT_COUNT%> CNDD1; "
            "float<%OPT_COUNT%> K2; float<%OPT_COUNT%> CNDD2; float<%OPT_COUNT%> expRT;"
            "float<%OPT_COUNT%> tCall; float<%OPT_COUNT%> tPut;\n"
            "    S = vload<%OPT_COUNT%>(opt, d_S);\n"
            "    X = vload<%OPT_COUNT%>(opt, d_X);\n"
            "    T = vload<%OPT_COUNT%>(opt, d_T);\n"
            "\n"
            "    sqrtT = SQRT(T);\n"
            "       d1 = (LOG(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);\n"
            "       d2 = d1 - V * sqrtT;\n"
            "        K = 1.0f / (1.0f + 0.2316419f * fabs(d1));\n"
            "\n"
            "    CNDD1 = RSQRT2PI * EXP(- 0.5f * d1 * d1) *\n"
            "                  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));\n"
            "\n"
            "<%IFD1%>"
            "\n"
            "    K2 = 1.0f / (1.0f + 0.2316419f * fabs(d2));\n"
            "\n"
            "    CNDD2 = RSQRT2PI * EXP(- 0.5f * d2 * d2) *\n"
            "                  (K2 * (A1 + K2 * (A2 + K2 * (A3 + K2 * (A4 + K2 * A5)))));\n"
            "\n"
            "<%IFD2%>"
            "    //Calculate Call and Put simultaneously\n"
            "    expRT = EXP(- R * T);\n"
            "    tCall = (S * CNDD1 - X * expRT * CNDD2);\n"
            "    tPut = (X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1));\n"
            /*"    printf(\"opt = <%d,%d>\\n\", opt, opt + 1);\n"
            "    printf(\"v = <%d,%d>\\n\", d_Call[opt], d_Call[opt + 1]);\n"*/
            "    vstore<%OPT_COUNT%>(tCall, opt, d_Call);\n"
            "    vstore<%OPT_COUNT%>(tPut, opt, d_Put);\n";



        std::string * loop_sReplaced = 0;
        std::string * ifd1_s = new std::string();
        std::string * ifd2_s = new std::string();

        std::string opt_count_s = isa::utils::toString(conf.getLoopUnrolling() + 1);
        loop_sReplaced = isa::utils::replace(&loop_sDecls, "<%OPT_COUNT%>", opt_count_s);

        std::string opt_size_s = isa::utils::toString(conf.getLoopUnrolling());
        std::string t = loop_sReplaced->c_str();
        loop_sReplaced = isa::utils::replace(&t, "<%OPT_SIZE%>", opt_size_s);


        for(unsigned int i = 0; i <= conf.getLoopUnrolling(); i++) {
          std::vector<std::string> identifier = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "sA", "sB", "sC", "sD", "sE", "sF"};
          std::string * temp = 0;
          std::string opt_s = isa::utils::toString(identifier[i]);
          std::string ifd1_sTemplate =  "    if(d1.<%OPT%> > 0)\n"
                                        "      CNDD1.<%OPT%> = 1.0f - CNDD1.<%OPT%>;\n";
          std::string ifd2_sTemplate =  "    if(d2.<%OPT%> > 0)\n"
                                        "      CNDD2.<%OPT%> = 1.0f - CNDD2.<%OPT%>;\n";


          temp = isa::utils::replace(&ifd1_sTemplate, "<%OPT%>", opt_s);
          ifd1_s->append(*temp);
          temp = isa::utils::replace(&ifd2_sTemplate, "<%OPT%>", opt_s);
          ifd2_s->append(*temp);
          delete temp;
        }
        std::string * temp = 0;
        std::string * loop_s = new std::string();
        loop_s->append(*loop_sReplaced);
        std::string t1 = loop_s->c_str();
        loop_s = isa::utils::replace(&t1, "<%IFD1%>", *ifd1_s);

        std::string t2 = loop_s->c_str();
        loop_s = isa::utils::replace(&t2, "<%IFD2%>", *ifd2_s);

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

