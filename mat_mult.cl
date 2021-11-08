__kernel void mat_mult_naive(
  const int N,
  __global float *A,
  __global float *B,
  __global float *C
) {
  int i=get_global_id(0);
  int j=get_global_id(1);
  C[i*N+j] = 0.0;
  for(int k=0; k<N; ++k) {
    C[i*N+j] += A[i*N+k]*B[k*N+j];
  }
}

__kernel void mat_mult_local_var(
  const int N,
  __global float *A,
  __global float *B,
  __global float *C
) {
  int i=get_global_id(0);
  int j=get_global_id(1);
  // Use local accumulator
  float tmp = 0.0;
  for(int k=0; k<N; ++k) {
    tmp += A[i*N+k]*B[k*N+j];
  }
  C[i*N+j] = tmp;
}

__kernel void mat_mult_1d(
  const int N,
  __global float *A,
  __global float *B,
  __global float *C
) {
  int i=get_global_id(0);
  // Each thread handles a column of B
  for(int j=0; j<N; ++j) {
    float tmp = 0.0;
    for(int k=0; k<N; ++k) {
      tmp += A[i*N+k]*B[k*N+j];
    }
    C[i*N+j] = tmp;
  }
}

__kernel void mat_mult_1d_col_copy(
  const int N,
  __global float *A,
  __global float *B,
  __global float *C,
  __local float *Bwrk
) {
  int i = get_global_id(0);
  int iloc = get_local_id(0);
  int nloc = get_local_size(0);

  for(int j=0; j<N; ++j) {
    // Cache a copy of B
    for(int k=iloc; k<N; k+=nloc) {
      Bwrk[k] = B[k*N+j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float tmp = 0.0;
    for(int k=0; k<N; ++k) {
      tmp += A[i*N+k]*Bwrk[k];
    }
    C[i*N+j] = tmp;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
}
