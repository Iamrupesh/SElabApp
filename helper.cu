#include <cuda.h>

extern "C" {
// transposes a column major matrix (i.e. &A[i][j] = A + i + (j * in_pitch)
//   to a column major output (&B[j][i] = B + j + (i * out_pitch), with
// i in [0, height-1], j in [0, width-1]
__global__ void transpose(void *dst, const void *src, int width, int height, int in_pitch, int out_pitch, int elem_size)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if((j >= width) || (i >= height)) return;

  char *dst_offset = (char *)dst;
  dst_offset += (j + i * out_pitch) * elem_size;

  const char *src_offset = (const char *)src;
  src_offset += (i + j * in_pitch) * elem_size;

  for(int k = 0; k < elem_size; k++) *dst_offset++ = *src_offset++;
}

texture <int2,1,cudaReadModeElementType> Atex;

__global__ void dtrsm_gpu_64_mm(double *A, double *B, int N, int texoff, int lda, int ldb)
{
  __shared__ double  As[17][17];
  __shared__ double  Bs[64][17];
  double temp;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

// Linear id of the thread(tx,ty)
  int tt = tx + blockDim.x*ty;

  int Boffset=ldb*blockDim.x*blockIdx.x;

  int uplim= ( blockDim.x *(blockIdx.x+1) > N) ?  (N-blockDim.x*blockIdx.x) : blockDim.x;

// Only the first 64 threads are loading a column at the time

  if (tt <64)
  {
 
// Each block will load 64 column of the RHS matrix, 16 rows at the time

   for ( int nn=0; nn<uplim; nn++)
   Bs[tt][nn] = *(B + nn *ldb +tt +Boffset );

   for ( int nn=uplim; nn<blockDim.x; nn++)
   Bs[tt][nn] = 0.;
  }

// Wait for all the loads in shared memory

   syncthreads();


// Perform triangular solve using 16x16  configuration
  int offset=0;
  for ( int n=0; n<4; n++)
  {
    offset=n*16;

    // Solve 16x16 block
    for( int ll=0; ll<16; ll++)
    {
      temp=Bs[ll+offset][tx];
      if (ty >ll)
       {
         int2 A1=tex1Dfetch(Atex,(texoff+(ll+offset) *lda +ty+offset));
         Bs[ty+offset   ][tx] -= __hiloint2double(A1.y,A1.x)*temp;
         //Bs[ty+offset   ][tx] -= A[(texoff+(ll+offset) *lda +ty+offset)]*temp;
       }
    syncthreads();
    }


   // Propagate the update
   for ( int k=n+1; k<4; k++)
   {
    int koffset=k*16;

    // syncthreads();
    int2 A1=tex1Dfetch(Atex,(texoff+(tx+offset) *lda +(ty+koffset)));
    As[ty][tx] = __hiloint2double(A1.y,A1.x);
    //As[ty][tx] = A[(texoff+(tx+offset) *lda +(ty+koffset))];
    syncthreads();

    double sum=0.;
   #pragma unroll 
    for (int kk=0;kk<16;kk++) 
      {
       //int2 A1=tex1Dfetch(Atex,((kk+offset) *lda +(ty+koffset)));
        //sum += _hiloint2double(A1.y,A1.x)*Bs[kk+offset][tx];
        sum += As[ty][kk]*Bs[kk+offset][tx];
      }
      Bs[ty+koffset][tx]-=sum;
    //syncthreads();
   }
    syncthreads();
  }

  if (tt <64)
  {
  for ( int nn=0; nn<uplim; nn++)
    *(B + nn *ldb +tt +Boffset )= Bs [tt][nn];
  }

  /*
     int woff=+tx *ldb +ty +Boffset;
    *( B+ woff )= Bs [ty][tx];
    *( B+ woff +16 )= Bs [ty+16][tx];
    *( B+ woff +32 )= Bs [ty+32][tx];
    *( B+ woff + 48 )= Bs [ty+48][tx];
  */

}

__global__ void dtrsm_gpu_64_mm_RT(double *A, double *B, int N, int texoff, int lda, int ldb)
{
  __shared__ double  As[17][17];
  __shared__ double  Bs[64][17];
  double temp;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Boffset=blockDim.x*blockIdx.x;

  int uplim= ( blockDim.x *(blockIdx.x+1) > N) ?  (N-blockDim.x*blockIdx.x) : blockDim.x;

    Bs[ty   ][tx] = 0.0;
    Bs[ty+16][tx] = 0.0;
    Bs[ty+32][tx] = 0.0;
    Bs[ty+48][tx] = 0.0;

  if(tx<uplim)
  {
    Bs[ty   ][tx] = *(B + (ty   )*ldb + tx + Boffset);
    Bs[ty+16][tx] = *(B + (ty+16)*ldb + tx + Boffset);
    Bs[ty+32][tx] = *(B + (ty+32)*ldb + tx + Boffset);
    Bs[ty+48][tx] = *(B + (ty+48)*ldb + tx + Boffset);
  }
// // Only the first 64 threads are loading a column at the time
// 
//   if (tt <64)
//   {
//  
// // Each block will load 64 column of the RHS matrix, 16 rows at the time
//    for ( int nn=0; nn<uplim; nn++)
//    Bs[tt][nn] = *(B + nn *ldb +tt +Boffset );
// 
//    for ( int nn=uplim; nn<blockDim.x; nn++)
//    Bs[tt][nn] = 0.;
//   }

// Wait for all the loads in shared memory

   syncthreads();


// Perform triangular solve using 16x16  configuration
  int offset=0;
  for ( int n=0; n<4; n++)
  {
    offset=n*16;

    // Solve 16x16 block
    for( int ll=0; ll<16; ll++)
    {
      temp=Bs[ll+offset][tx];
      if (ty >ll)
       {
         int2 A1=tex1Dfetch(Atex,(texoff+(ll+offset) *lda +ty+offset));
         Bs[ty+offset   ][tx] -= __hiloint2double(A1.y,A1.x)*temp;
       }
    syncthreads();
    }


   // Propagate the update
   for ( int k=n+1; k<4; k++)
   {
    int koffset=k*16;

    // syncthreads();
    int2 A1=tex1Dfetch(Atex,(texoff+(tx+offset) *lda +(ty+koffset)));
    As[ty][tx] = __hiloint2double(A1.y,A1.x);
    syncthreads();

    double sum=0.;
   #pragma unroll 
    for (int kk=0;kk<16;kk++) 
      {
       //int2 A1=tex1Dfetch(Atex,((kk+offset) *lda +(ty+koffset)));
        //sum += _hiloint2double(A1.y,A1.x)*Bs[kk+offset][tx];
        sum += As[ty][kk]*Bs[kk+offset][tx];
      }
      Bs[ty+koffset][tx]-=sum;
    //syncthreads();
   }
    syncthreads();
  }

  if(tx<uplim)
  {
    *(B + (ty   )*ldb + tx + Boffset) = Bs[ty   ][tx];
    *(B + (ty+16)*ldb + tx + Boffset) = Bs[ty+16][tx];
    *(B + (ty+32)*ldb + tx + Boffset) = Bs[ty+32][tx];
    *(B + (ty+48)*ldb + tx + Boffset) = Bs[ty+48][tx];
  }

//   if (tt <64)
//   {
//   for ( int nn=0; nn<uplim; nn++)
//     *(B + nn *ldb +tt +Boffset )= Bs [tt][nn];
//   }

  /*
     int woff=+tx *ldb +ty +Boffset;
    *( B+ woff )= Bs [ty][tx];
    *( B+ woff +16 )= Bs [ty+16][tx];
    *( B+ woff +32 )= Bs [ty+32][tx];
    *( B+ woff + 48 )= Bs [ty+48][tx];
  */

}


};
