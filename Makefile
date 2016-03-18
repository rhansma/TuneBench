
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


all: bin/Reduction.o bin/ReductionTuner

bin/Reduction.o: $(UTILS)/bin/utils.o include/Reduction.hpp src/Reduction.cpp
	$(CC) -o bin/Reduction.o -c src/Reduction.cpp $(CL_INCLUDES) $(CFLAGS)

bin/ReductionTuner: $(CL_DEPS) bin/Reduction.o src/ReductionTuner.cpp
	$(CC) -o bin/ReductionTuner src/ReductionTuner.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

