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

#if(1)
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
}
