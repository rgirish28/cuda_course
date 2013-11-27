###########################################################################################
### 		Compiler Setup
###########################################################################################


# The GPU architecture you are building for, e.g. 13, 20
ARCH ?= 20

ASTYLE := ~/astyle --style=kr --indent=spaces=4

# Change compiler flags as needed
BUILDFLAGS := -O2 -arch compute_$(ARCH) -code sm_$(ARCH) -m 64 --ptxas-options=-v -Xcompiler -fopenmp
BUILDFLAGS += -I ./ -I $(NAGGPU)/include -L $(NAGGPU)/lib -lnag_gpu

# Set path to nvcc
#NVCCDIR := /fserver/gpubuild/buildtools/redhat/NVIDIA/5.0
NVCC := nvcc


# by default build the examples
.DEFAULT_GOAL := tutorial


###########################################################################################
## 
##       Build Rules
##
###########################################################################################


tutorial :
	$(NVCC) $(BUILDFLAGS) tut4_multigpu.cu -o tut4.exe


solution :
	$(NVCC) $(BUILDFLAGS) tut4_multigpu_solution.cu -o tut4_solution.exe

clean :
	rm -f tut4.exe tut4_solution.exe
