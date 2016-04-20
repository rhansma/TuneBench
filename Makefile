
# https://github.com/isazi/utils
UTILS := $(HOME)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(HOME)/src/OpenCL

INCLUDES := -I"include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL

CC := g++

# Dependencies
DEPS := $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 


all: bin/Reduction.o bin/ReductionTuner bin/ReductionPrint bin/Stencil.o bin/StencilTuner bin/StencilPrint bin/MD.o bin/MDTuner bin/MDPrint bin/TriadTuner bin/TriadPrint bin/Correlator.o bin/CorrelatorPrint

bin/Reduction.o: $(UTILS)/bin/utils.o include/Reduction.hpp src/Reduction.cpp
	$(CC) -o bin/Reduction.o -c src/Reduction.cpp $(CL_INCLUDES) $(CFLAGS)

bin/ReductionTuner: $(CL_DEPS) bin/Reduction.o include/configuration.hpp src/ReductionTuner.cpp
	$(CC) -o bin/ReductionTuner src/ReductionTuner.cpp bin/Reduction.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/ReductionPrint: $(CL_DEPS) bin/Reduction.o include/configuration.hpp src/ReductionPrint.cpp
	$(CC) -o bin/ReductionPrint src/ReductionPrint.cpp bin/Reduction.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Stencil.o: $(UTILS)/bin/utils.o include/Stencil.hpp src/Stencil.cpp
	$(CC) -o bin/Stencil.o -c src/Stencil.cpp $(CL_INCLUDES) $(CFLAGS)

bin/StencilTuner: $(CL_DEPS) bin/Stencil.o include/configuration.hpp src/StencilTuner.cpp
	$(CC) -o bin/StencilTuner src/StencilTuner.cpp bin/Stencil.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/StencilPrint: $(CL_DEPS) bin/Stencil.o include/configuration.hpp src/StencilPrint.cpp
	$(CC) -o bin/StencilPrint src/StencilPrint.cpp bin/Stencil.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/MD.o: $(UTILS)/bin/utils.o include/MD.hpp src/MD.cpp
	$(CC) -o bin/MD.o -c src/MD.cpp $(CL_INCLUDES) $(CFLAGS)

bin/MDTuner: $(CL_DEPS) bin/MD.o include/configuration.hpp src/MDTuner.cpp
	$(CC) -o bin/MDTuner src/MDTuner.cpp bin/MD.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/MDPrint: $(CL_DEPS) bin/MD.o include/configuration.hpp src/MDPrint.cpp
	$(CC) -o bin/MDPrint src/MDPrint.cpp bin/MD.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/TriadTuner: $(CL_DEPS) include/configuration.hpp include/Triad.hpp src/TriadTuner.cpp
	$(CC) -o bin/TriadTuner src/TriadTuner.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/TriadPrint: $(CL_DEPS) include/configuration.hpp include/Triad.hpp src/TriadPrint.cpp
	$(CC) -o bin/TriadPrint src/TriadPrint.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Correlator.o: $(UTILS)/bin/utils.o include/Correlator.hpp src/Correlator.cpp
	$(CC) -o bin/Correlator.o -c src/Correlator.cpp $(CL_INCLUDES) $(CFLAGS)

bin/CorrelatorPrint: $(CL_DEPS) bin/Correlator.o include/configuration.hpp src/CorrelatorPrint.cpp
	$(CC) -o bin/CorrelatorPrint src/CorrelatorPrint.cpp bin/Correlator.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

