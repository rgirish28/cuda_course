###########################################################################################
### 		Compiler Setup
###########################################################################################


# The GPU architecture you are building for, e.g. 13, 20
ARCH ?= 20


# Change compiler flags as needed
BUILDFLAGS := -O2 -arch sm_$(ARCH) -m 64 --ptxas-options=-v


# Set path to nvcc
NVCCDIR ?= /fserver/gpubuild/buildtools/redhat/NVIDIA/5.0
#NVCC := $(NVCCDIR)/bin/nvcc
NVCC := nvcc


# by default build the examples
.DEFAULT_GOAL := tutorial


###########################################################################################
## 
##       Build Rules
##
###########################################################################################
NAG_FLAGS ?= -L $(NAGGPU)/lib -I $(NAGGPU)/include -l nag_gpu



tutorial :
	$(NVCC) $(NAG_FLAGS) $(BUILDFLAGS) tut1_monteCarlo.cu -o tut1.exe


solution :
	$(NVCC) $(NAG_FLAGS) $(BUILDFLAGS) tut1_monteCarlo_solution.cu -o tut1_solution.exe

clean :
	rm -f tut1.exe tut1_solution.exe
