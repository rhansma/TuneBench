
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


all: bin/Reduction.o bin/ReductionTuner bin/Stencil.o bin/StencilTuner

bin/Reduction.o: $(UTILS)/bin/utils.o include/Reduction.hpp src/Reduction.cpp
	$(CC) -o bin/Reduction.o -c src/Reduction.cpp $(CL_INCLUDES) $(CFLAGS)

bin/ReductionTuner: $(CL_DEPS) bin/Reduction.o include/configuration.hpp src/ReductionTuner.cpp
	$(CC) -o bin/ReductionTuner src/ReductionTuner.cpp bin/Reduction.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/Stencil.o: $(UTILS)/bin/utils.o include/Stencil.hpp src/Stencil.cpp
	$(CC) -o bin/Stencil.o -c src/Stencil.cpp $(CL_INCLUDES) $(CFLAGS)

bin/StencilTuner: $(CL_DEPS) bin/Stencil.o include/configuration.hpp src/StencilTuner.cpp
	$(CC) -o bin/StencilTuner src/StencilTuner.cpp bin/Stencil.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/MD.o: $(UTILS)/bin/utils.o include/MD.hpp src/MD.cpp
	$(CC) -o bin/MD.o -c src/MD.cpp $(CL_INCLUDES) $(CFLAGS)

bin/MDTuner: $(CL_DEPS) bin/MD.o include/configuration.hpp src/MDTuner.cpp
	$(CC) -o bin/MDTuner src/MDTuner.cpp bin/MD.o $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

