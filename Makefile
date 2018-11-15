###########################################################
#  Makefile for Fermi DGEMM/DTRSM library                 #
###########################################################


ifndef version
version=4
endif

CUBIN=dgemm_kernels_sass_v$(version).cubin
DEFINES=-DVERSION$(version)

all : dgemm_test 

dgemm_test : dgemm_test.c fermi_dgemm.o 
	gcc dgemm_test.c fermi_dgemm.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcudart -lcuda -o dgemm_test

fermi_dgemm.o : fermi_dgemm.c fermi_dgemm.h helper_nvcc.cubin $(CUBIN)
	gcc -O0 $(DEFINES) -c fermi_dgemm.c -o fermi_dgemm.o -I/usr/local/cuda/include

%_nvcc.cubin : %.cu
	nvcc --ptxas-options=-v -arch sm_20 -cubin $^ -o $@ -I/usr/local/cuda/include

clean:
	rm -f dgemm_test fermi_dgemm.o 
