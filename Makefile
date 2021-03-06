
# https://github.com/isazi/utils
UTILS := $(HOME)/git/scriptie/utils
# https://github.com/isazi/OpenCL
OPENCL := $(HOME)/git/scriptie/OpenCL

OPENCL_HEADERS := $(HOME)/git/OpenCL-CLHPP/build/include
OPENCL_HEADERS2 := $(HOME)/NVIDIASDK/OpenCL/common/inc

INCLUDES := -I"include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include" -I"$(OPENCL_HEADERS)" -I"$(OPENCL_HEADERS2)"
CL_LIBS := -L"$(OPENCL_LIB)"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0 -fopenmp
else
	CFLAGS += -O0 -g3
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL

CC := g++

# Dependencies
DEPS := $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 


all: bin/BlackScholes.o bin/BlackScholesPrint bin/Reduction.o bin/ReductionPrint bin/Stencil.o bin/StencilPrint bin/MD.o bin/MDPrint bin/Triad.o bin/TriadPrint bin/Correlator.o bin/CorrelatorPrint bin/Tuner

bin/BlackScholes.o: $(UTILS)/bin/utils.o include/BlackScholes.hpp src/BlackScholes.cpp
	$(CC) -o bin/BlackScholes.o -c src/BlackScholes.cpp $(CL_INCLUDES) $(CFLAGS)

bin/BlackScholesPrint: $(CL_DEPS) bin/BlackScholes.o include/configuration.hpp src/BlackScholesPrint.cpp
	$(CC) -o bin/BlackScholesPrint src/BlackScholesPrint.cpp bin/BlackScholes.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Reduction.o: $(UTILS)/bin/utils.o include/Reduction.hpp src/Reduction.cpp
	$(CC) -o bin/Reduction.o -c src/Reduction.cpp $(CL_INCLUDES) $(CFLAGS)

bin/ReductionPrint: $(CL_DEPS) bin/Reduction.o include/configuration.hpp src/ReductionPrint.cpp
	$(CC) -o bin/ReductionPrint src/ReductionPrint.cpp bin/Reduction.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Stencil.o: $(UTILS)/bin/utils.o include/Stencil.hpp src/Stencil.cpp
	$(CC) -o bin/Stencil.o -c src/Stencil.cpp $(CL_INCLUDES) $(CFLAGS)

bin/StencilPrint: $(CL_DEPS) bin/Stencil.o include/configuration.hpp src/StencilPrint.cpp
	$(CC) -o bin/StencilPrint src/StencilPrint.cpp bin/Stencil.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/MD.o: $(UTILS)/bin/utils.o include/MD.hpp src/MD.cpp
	$(CC) -o bin/MD.o -c src/MD.cpp $(CL_INCLUDES) $(CFLAGS)

bin/MDPrint: $(CL_DEPS) bin/MD.o include/configuration.hpp src/MDPrint.cpp
	$(CC) -o bin/MDPrint src/MDPrint.cpp bin/MD.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Triad.o: include/Triad.hpp src/Triad.cpp
	$(CC) -o bin/Triad.o -c src/Triad.cpp $(CL_INCLUDES) $(CFLAGS)

bin/TriadPrint: $(CL_DEPS) include/configuration.hpp bin/Triad.o src/TriadPrint.cpp
	$(CC) -o bin/TriadPrint src/TriadPrint.cpp bin/Triad.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Correlator.o: $(UTILS)/bin/utils.o include/Correlator.hpp src/Correlator.cpp
	$(CC) -o bin/Correlator.o -c src/Correlator.cpp $(CL_INCLUDES) $(CFLAGS)

bin/CorrelatorPrint: $(CL_DEPS) bin/Correlator.o include/configuration.hpp src/CorrelatorPrint.cpp
	$(CC) -o bin/CorrelatorPrint src/CorrelatorPrint.cpp bin/Correlator.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Tuner: $(CL_DEPS) bin/BlackScholes.o bin/Reduction.o bin/Correlator.o bin/Stencil.o bin/Triad.o bin/MD.o include/StringToArgcArgv.hpp include/configuration.hpp include src/Tuner.cpp
	$(CC) -o bin/Tuner src/Tuner.cpp src/BlackScholesTuner.cpp bin/BlackScholes.o src/ReductionTuner.cpp bin/Reduction.o src/CorrelatorTuner.cpp bin/Correlator.o src/MDTuner.cpp bin/MD.o src/StencilTuner.cpp bin/Stencil.o src/TriadTuner.cpp bin/Triad.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

