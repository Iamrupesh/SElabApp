// hand-coded DGEMM kernels for Fermi

// intended to work just like cublas (i.e. runtime-style parameters), although only for a subset of 
//   the parameter space

#ifndef FERMI_DGEMM_H
#define FERMI_DGEMM_H

// creates one or more streams for use with other fermiDgemm/fermi{Get,Set}Matrix calls
// streams will be numbered 1 .. 'count'
// returns 0 on success
int fermiCreateStreams(int count);
int fermiSyncStream(int stream);

// memcpy's from host2dev and dev2host using a given stream - otherwise identical to 
// cublas{Get,Set}Matrix
int fermiSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream);
int fermiGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, int stream);

// returns 0 on success
// non-zero return means nth parameter (starting counting at 1) has an unsupported value
int fermiDgemm(char transa, char transb, int m, int n, int k,
               double alpha, const double *A, int lda, 
               const double *B, int ldb, double beta, double *C, 
               int ldc);

int fermiDgemm_stream(char transa, char transb, int m, int n, int k,
                      double alpha, const double *A, int lda, 
                      const double *B, int ldb, double beta, double *C, 
                      int ldc, int stream);

int fermi_transpose(double *dst, double *src, int width, int height, int in_pitch, int out_pitch, int stream );

int dtrsm_gpu( char side, char uplo, char transa,
               char diag, int M, int N, double alpha,
               double *A, int lda, double *B, int ldb, int stream);

#endif
