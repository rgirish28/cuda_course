###########################################################################################
### 		Compiler Setup
###########################################################################################


# The GPU architecture you are building for, e.g. 13, 20
ARCH ?= 20


# Change compiler flags as needed
BUILDFLAGS := -O2 -arch compute_$(ARCH) -code sm_$(ARCH) -m 64 --ptxas-options=-v -Xcompiler -fopenmp
BUILDFLAGS += -L $(NAGGPU)/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -liomp5 

# Set path to nvcc
NVCCDIR := /fserver/gpubuild/buildtools/redhat/NVIDIA/5.0
#NVCC := $(NVCCDIR)/bin/nvcc
NVCC := nvcc


# by default build the examples
.DEFAULT_GOAL := tutorial


###########################################################################################
## 
##       Build Rules
##
###########################################################################################


tutorial :
	$(NVCC) $(BUILDFLAGS) tut3_concurrency.cu -o tut3.exe


solution :
	$(NVCC) $(BUILDFLAGS) tut3_concurrency_solution.cu -o tut3_solution.exe

clean :
	rm -f tut3.exe tut3_solution.exe
