#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#define EPSILON 0.0001

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <chrono>
using namespace std::chrono;

int setDefaultPlatform(const std::string& targetName) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform plat;

  // Print out names of all available platforms
  std::cout << "Available platforms:" << std::endl;
  for (auto &p : platforms) {
    std::string platname = p.getInfo<CL_PLATFORM_NAME>();
    std::cout << platname << std::endl;
  }

  // Find first platform matching name
  for (auto &p : platforms) {
    std::string platname = p.getInfo<CL_PLATFORM_NAME>();
    if (platname.find(targetName) != std::string::npos) {
      plat = p;
      break;
    }
  }

  if (plat() == 0)  {
    std::cout << "No platform found with name " << targetName << ".\n";
    return -1;
  }

  cl::Platform newP = cl::Platform::setDefault(plat);
  if (newP != plat) {
    std::cout << "Error setting default platform.";
    return -1;
  }

  std::string platname = newP.getInfo<CL_PLATFORM_NAME>();
  std::cout << "Running on " << platname << std::endl;

  return 0;
}

auto read_file(std::string_view path) -> std::string {
  // Read entire file into string
  // stolen from https://stackoverflow.com/a/116220
  constexpr auto read_size = std::size_t{4096};
  auto stream = std::ifstream{path.data()};
  stream.exceptions(std::ios_base::badbit);

  auto out = std::string{};
  auto buf = std::string(read_size, '\0');
  while (stream.read(& buf[0], read_size)) {
    out.append(buf, 0, stream.gcount());
  }
  out.append(buf, 0, stream.gcount());
  return out;
}

cl::Program buildProgram(const std::string& filename) {
  // Compile kernel source into program
  cl::Program program(read_file(filename), false);
  try {
    program.build("-cl-std=CL2.0");
  }
  catch (...) {
    std::string bl = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault());
    std::cerr << bl << std::endl;
  }

  return program;
}

inline int idx(int i, int j, int N) {
  // Correctly calculate the global index 100% of the time, every time
  return i*N+j;
}

void mat_mult(int N, float *A, float *B, float *C) {
  // Basic CPU implementation
  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      float tmp=0.0;
      for(int k=0; k<N; ++k) {
        // C(i,j) = sum_k A(i,k)*B(k,i)
        tmp += A[idx(i,k,N)]*B[idx(k,j,N)];
      }
      C[idx(i,j,N)] = tmp;
    }
  }
}

void mat_mult_omp(int N, float *A, float *B, float *C) {
  // Basic OpenMP implementation
#pragma omp parallel for collapse(2)
  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      float tmp=0.0;
      for(int k=0; k<N; ++k) {
        // C(i,j) = sum_k A(i,k)*B(k,i)
        tmp += A[idx(i,k,N)]*B[idx(k,j,N)];
      }
      C[idx(i,j,N)] = tmp;
    }
  }
}

void load_random_data(std::vector<float>& A_in, std::vector<float>& B_in) {
  // Fill the matrices with essentially random integers (distributed around 0)
  for(int i=0; i<A_in.size(); ++i) {
    A_in[i] = i%11-5;
    B_in[i] = (i-5)%11-5;
  }
}

void load_test_case(std::vector<float>& A_in, std::vector<float>& B_in) {
  // Load specific test data
  float A[] = {
    5,  6,  3,
    7,  2, -2,
    4, -1,  8
  };

  float B[] = {
    -3, 0, 1,
     3, 5, 6,
    -2, 4, 7
  };

  for (int i=0; i<3*3; ++i) {
    A_in[i] = A[i];
    B_in[i] = B[i];
  }
}

void check_test_result(std::vector<float>& C_in) {
  // Check input against known result
  float C[] = {
    -3, 42, 62,
    -11, 2, 5,
    -31, 27, 54
  };

  for(int i=0; i<3*3; ++i) {
    assert(C[i] == C_in[i]);
  }
}

void verify_seq() {
  // Make sure basic CPU implementation works
  const int N = 3;

  std::vector<float> h_A(N*N), h_B(N*N), h_C(N*N);
  load_test_case(h_A, h_B);

  mat_mult(N, h_A.data(), h_B.data(), h_C.data());

  check_test_result(h_C);
}

void run_seq(
    const int N,
    std::vector<float>& h_A,
    std::vector<float>& h_B,
    std::vector<float>& h_C)
{
  // Time sequential
  auto start = high_resolution_clock::now();

  mat_mult(N, h_A.data(), h_B.data(), h_C.data());

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  double nflop = 2.0*N*N*N;
  std::cout << "N=" << N << ", MFLOPS=" << int(nflop/float(duration.count())) << std::endl;
}

void run_seq_omp(
    const int N,
    std::vector<float>& h_A,
    std::vector<float>& h_B,
    std::vector<float>& h_C)
{
  // Time OpenMP
  auto start = high_resolution_clock::now();

  mat_mult_omp(N, h_A.data(), h_B.data(), h_C.data());

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  double nflop = 2.0*N*N*N;
  std::cout << "N=" << N << ", MFLOPS=" << int(nflop/float(duration.count())) << std::endl;
}

void verify_ocl(const std::string& kernelName) {
  // Ensure the basic OpenCL version works
  const int N = 3;

  // Host mem
  std::vector<float> h_A(N*N), h_B(N*N), h_C(N*N);
  load_test_case(h_A, h_B);

  // Device mem
  cl::Buffer d_A = cl::Buffer(h_A.begin(), h_A.end(), true);
  cl::Buffer d_B = cl::Buffer(h_B.begin(), h_B.end(), true);
  cl::Buffer d_C = cl::Buffer(CL_MEM_READ_WRITE, sizeof(float)*h_C.size());

  // Load program & extract kernel named kernelName
  cl::Program program = buildProgram("mat_mult.cl");
  auto mat_mult_cl = cl::KernelFunctor<
    int, cl::Buffer, cl::Buffer, cl::Buffer
    >(program, kernelName);

  // run kernel
  mat_mult_cl(cl::EnqueueArgs(cl::NDRange(N, N)), N, d_A, d_B, d_C);

  // Get data back
  cl::copy(d_C, h_C.begin(), h_C.end());

  // Check result
  check_test_result(h_C);
}

void run_ocl(
    const std::string& kernelName,
    const int N,
    std::vector<float>& h_A,
    std::vector<float>& h_B,
    std::vector<float>& h_C)
{
  // Time basic OpenCL
  cl::Buffer d_A = cl::Buffer(h_A.begin(), h_A.end(), true);
  cl::Buffer d_B = cl::Buffer(h_B.begin(), h_B.end(), true);
  cl::Buffer d_C = cl::Buffer(CL_MEM_READ_WRITE, sizeof(float)*h_C.size());

  cl::Program program = buildProgram("mat_mult.cl");
  auto mat_mult_cl = cl::KernelFunctor<
    int, cl::Buffer, cl::Buffer, cl::Buffer
    >(program, kernelName);

  auto start = high_resolution_clock::now();

  mat_mult_cl(cl::EnqueueArgs(cl::NDRange(N, N)), N, d_A, d_B, d_C);

  cl::copy(d_C, h_C.begin(), h_C.end());

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  double nflop = 2.0*N*N*N;
  std::cout << "N=" << N << ", MFLOPS=" << int(nflop/float(duration.count())) << std::endl;
}

void run_1d_ocl(
    const std::string& kernelName,
    const int N,
    std::vector<float>& h_A,
    std::vector<float>& h_B,
    std::vector<float>& h_C)
{
  cl::Buffer d_A = cl::Buffer(h_A.begin(), h_A.end(), true);
  cl::Buffer d_B = cl::Buffer(h_B.begin(), h_B.end(), true);
  cl::Buffer d_C = cl::Buffer(CL_MEM_READ_WRITE, sizeof(float)*h_C.size());

  cl::Program program = buildProgram("mat_mult.cl");
  auto mat_mult_cl = cl::KernelFunctor<
    int, cl::Buffer, cl::Buffer, cl::Buffer
    >(program, kernelName);

  auto start = high_resolution_clock::now();
  // Run over 1 dimension, use workgroups of N/16
  mat_mult_cl(cl::EnqueueArgs(cl::NDRange(N), cl::NDRange(N/16)), N, d_A, d_B, d_C);

  cl::copy(d_C, h_C.begin(), h_C.end());

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  double nflop = 2.0*N*N*N;
  std::cout << "N=" << N << ", MFLOPS=" << int(nflop/float(duration.count())) << std::endl;
}

void run_1d_wrk_ocl(
    const std::string& kernelName,
    const int N,
    std::vector<float>& h_A,
    std::vector<float>& h_B,
    std::vector<float>& h_C)
{
  cl::Buffer d_A = cl::Buffer(h_A.begin(), h_A.end(), true);
  cl::Buffer d_B = cl::Buffer(h_B.begin(), h_B.end(), true);
  cl::Buffer d_C = cl::Buffer(CL_MEM_READ_WRITE, sizeof(float)*h_C.size());

  // Allocate temp space local to workgroups
  cl::LocalSpaceArg d_wrk = cl::Local(sizeof(float)*N);

  cl::Program program = buildProgram("mat_mult.cl");
  // Build kernel with extra workgroup space
  auto mat_mult_cl = cl::KernelFunctor<
    int, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg
    >(program, kernelName);

  auto start = high_resolution_clock::now();

  // Pass in workgroup
  mat_mult_cl(cl::EnqueueArgs(cl::NDRange(N), cl::NDRange(N/4)), N, d_A, d_B, d_C, d_wrk);

  cl::copy(d_C, h_C.begin(), h_C.end());

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  double nflop = 2.0*N*N*N;
  std::cout << "N=" << N << ", MFLOPS=" << int(nflop/float(duration.count())) << std::endl;
}

void check_equal(const std::vector<float>& a, const std::vector<float>& b) {
  for (int i=0; i<a.size(); ++i) {
    //std::cout << a[i] << ", " << b[i] << std::endl;
    assert(std::abs(a[i]-b[i]) < EPSILON);
  }
}

int main() {
  // Setup OpenCL
  //int error = setDefaultPlatform("Intel(R) OpenCL HD Graphics");
  int error = setDefaultPlatform("Intel(R) CPU Runtime");
  if (error < 0) return -1;

  cl::DeviceCommandQueue deviceQueue = cl::DeviceCommandQueue::makeDefault(
      cl::Context::getDefault(), cl::Device::getDefault());

  verify_seq();

  std::vector<int> Ns = {512};

  for (auto N : Ns) {
    std::vector<float> h_A(N*N), h_B(N*N), h_C(N*N), h_C_seq(N*N);
    load_random_data(h_A, h_B);

    std::cout << "Seq:   ";
    run_seq(N, h_A, h_B, h_C_seq);

    std::cout << "OMP:   ";
    run_seq_omp(N, h_A, h_B, h_C);
    check_equal(h_C, h_C_seq);

    std::cout << "Naive: ";
    run_ocl("mat_mult_naive", N, h_A, h_B, h_C);
    check_equal(h_C, h_C_seq);

    std::cout << "Local: ";
    run_ocl("mat_mult_local_var", N, h_A, h_B, h_C);
    check_equal(h_C, h_C_seq);

    std::cout << "1D:    ";
    run_1d_ocl("mat_mult_1d", N, h_A, h_B, h_C);
    check_equal(h_C, h_C_seq);

    // this version does weird things, probably shouldn't bother
    //std::cout << "1D cc:  ";
    //run_1d_wrk_ocl("mat_mult_1d_col_copy", N, h_A, h_B, h_C);
    //check_equal(h_C, h_C_seq);
  }
}
