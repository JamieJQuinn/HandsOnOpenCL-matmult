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
  float tmp = 0.0;
  for(int k=0; k<N; ++k) {
    tmp += A[i*N+k]*B[k*N+j];
  }
  C[i*N+j] = tmp;
}
