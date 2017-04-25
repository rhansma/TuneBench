/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#if(0)
#define EXP(a) native_exp(a)
    #define LOG(a) native_log(a)
    #define SQRT(a) native_sqrt(a)
#else
#define EXP(a) exp(a)
#define LOG(a) log(a)
#define SQRT(a) sqrt(a)
#endif

__kernel void BlackScholes(
    __global float *d_Call, //Call option price
    __global float *d_Put,  //Put option price
    __global float *d_S,    //Current stock price
    __global float *d_X,    //Option strike price
    __global float *d_T,    //Option years
    float R,                //Riskless rate of return
    float V,                //Stock volatility
    unsigned int optN
){
  const float       A1 = 0.31938153f;
  const float       A2 = -0.356563782f;
  const float       A3 = 1.781477937f;
  const float       A4 = -1.821255978f;
  const float       A5 = 1.330274429f;
  const float RSQRT2PI = 0.39894228040143267793994605993438f;

  <%LOOP_UNROLL%>
  for(unsigned int opt = get_global_id(0); opt < optN; opt += get_global_size(0)) {
    float S = d_S[opt];
    float X = d_X[opt];
    float T = d_T[opt];

    float sqrtT = SQRT(T);
    float    d1 = (LOG(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    float    d2 = d1 - V * sqrtT;
    float
        K = 1.0f / (1.0f + 0.2316419f * fabs(d1));

    float CNDD1 = RSQRT2PI * EXP(- 0.5f * d1 * d1) *
              (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d1 > 0)
      CNDD1 = 1.0f - CNDD1;

    K = 1.0f / (1.0f + 0.2316419f * fabs(d2));

    float CNDD2 = RSQRT2PI * EXP(- 0.5f * d2 * d2) *
                  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d2 > 0)
      CNDD2 = 1.0f - CNDD2;

    //Calculate Call and Put simultaneously
    float expRT = EXP(- R * T);
    d_Call[opt] = (S * CNDD1 - X * expRT * CNDD2);
    d_Put[opt]  = (X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1));
  }
}
