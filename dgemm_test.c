#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "cublas.h"
#include "fermi_dgemm.h"

#define CUDA_CALL(x) do { cudaError_t res = (x); if(res != cudaSuccess) { fprintf(stderr, "%s: %d, %s\n", #x, res, cudaGetErrorString(res)); exit(1); } } while(0)

// lets get enough space for 256M doubles (1GB)
#define BUFFER_SIZE 300*1024*1024

static const unsigned mn_vals[] = { 512,1024,2048,3072,4096,8192 };
//static const unsigned mn_vals[] = { 4096, 8192 };

//static const unsigned k_vals[] = { 64,128,256,512,1024,2048,4096,8192,16384 };
static const unsigned k_vals[] = { 768, 1024, 2048, 4096, 8192 };

int main(int argc, const char *argv[])
{
  int dev;
  double *data;
  size_t free_bytes, total_bytes;
  cudaEvent_t ev_start, ev_end;
  struct cudaDeviceProp props;
  cudaSetDevice(3);
  CUDA_CALL( cudaGetDevice(&dev) );
  CUDA_CALL( cudaGetDeviceProperties(&props, dev) );
  CUDA_CALL( cudaMemGetInfo(&free_bytes, &total_bytes) );

  CUDA_CALL( cudaMalloc((void **)&data, BUFFER_SIZE * sizeof(double)) );
  //printf("Malloc returned %p\n", data);

  CUDA_CALL( cudaEventCreate(&ev_start) );
  CUDA_CALL( cudaEventCreate(&ev_end) );

  printf("# Running on '%s'\n", props.name);
  printf("# SMs = %d\n", props.multiProcessorCount);
  printf("# clock = %d\n", props.clockRate);
  printf("# memory = %ld (%ld free)\n", total_bytes, free_bytes);

  printf("atrans\tbtrans\tM\tN\tK\tgflops\tcublas\n");

  int mni, ki, btrans;

  for(btrans = 1; btrans < 2; btrans++) {
  for(mni = 0; mni < sizeof(mn_vals)/sizeof(int); mni++) {
    for(ki = 0; ki < sizeof(k_vals)/sizeof(int); ki++) {
      unsigned mn = mn_vals[mni];
      unsigned k = k_vals[ki];

      double *A, *B, *C;

      float cublas_ms, fermi_ms;
      float cublas_gflops, fermi_gflops;

      // do we have enough space?
      if(((mn * mn) + 2 * (k * mn)) > BUFFER_SIZE) continue;

      A = data;
      B = A + (k * mn);
      C = B + (k * mn);

#define TIME3(ms,func,args) do { \
      func args; \
      CUDA_CALL( cudaEventRecord(ev_start, 0) ); \
      func args; \
      CUDA_CALL( cudaEventRecord(ev_end, 0) ); \
      func args; \
      CUDA_CALL( cudaEventSynchronize(ev_end) ); \
      CUDA_CALL( cudaEventElapsedTime(&ms, ev_start, ev_end) ); \
   } while(0)

      // cublas first
      TIME3(cublas_ms, cublasDgemm, ('N', (btrans ? 'T' : 'N'), mn, mn, k, 0.5, A, mn, B, mn, 0.5, C, mn));
      TIME3(fermi_ms, fermiDgemm, ('N', (btrans ? 'T' : 'N'), mn, mn, k, 0.5, A, mn, B, mn, 0.5, C, mn));
#if 0
      cublasDgemm('N', 'T', mn, mn, k, 0.5, A, mn, B, k, 0.5, C, mn);
      CUDA_CALL( cudaEventRecord(ev_start, 0) );
      cublasDgemm('N', 'T', mn, mn, k, 0.5, A, mn, B, k, 0.5, C, mn);
      CUDA_CALL( cudaEventRecord(ev_end, 0) );
      cublasDgemm('N', 'T', mn, mn, k, 0.5, A, mn, B, k, 0.5, C, mn);
      CUDA_CALL( cudaEventSynchronize(ev_end) );
      CUDA_CALL( cudaEventElapsedTime(&cublas_ms, ev_start, ev_end) );

      // fermi next
      fermiDgemm('N', 'T', mn, mn, k, 0.5, A, mn, B, k, 0.5, C, mn);
      CUDA_CALL( cudaEventRecord(ev_start, 0) );
      fermiDgemm('N', 'T', mn, mn, k, 0.5, A, mn, B, k, 0.5, C, mn);
      CUDA_CALL( cudaEventRecord(ev_end, 0) );
      fermiDgemm('N', 'T', mn, mn, k, 0.5, A, mn, B, k, 0.5, C, mn);
      CUDA_CALL( cudaEventSynchronize(ev_end) );
      CUDA_CALL( cudaEventElapsedTime(&fermi_ms, ev_start, ev_end) );
#endif

      cublas_gflops = (1e-6 * mn * mn * (2 * k + 1)) / cublas_ms;
      fermi_gflops = (1e-6 * mn * mn * (2 * k + 1)) / fermi_ms;

      printf("%c\t%c\t%d\t%d\t%d\t%5.3f\t%5.3f\n",
             'N', (btrans ? 'T' : 'N'),
             mn, mn, k, fermi_gflops,cublas_gflops);
      fflush(stdout);
    }
  }
  }

  return 0;
}
