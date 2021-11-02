#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

int setDefaultPlatform() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform plat;
  for (auto &p : platforms) {
    std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
    if (platver.find("OpenCL 3.") != std::string::npos) {
      plat = p;
    }
  }

  if (plat() == 0)  {
    std::cout << "No OpenCL 3.0 platform found.\n";
    return -1;
  }

  cl::Platform newP = cl::Platform::setDefault(plat);
  if (newP != plat) {
    std::cout << "Error setting default platform.";
    return -1;
  }

  return 0;
}

auto read_file(std::string_view path) -> std::string {
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
  cl::Program program(read_file(filename), false);
  try {
    program.build("-cl-std=CL3.0");
  }
  catch (...) {
    std::string bl = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault());
    std::cerr << bl << std::endl;
  }

  return program;
}

inline int idx(int i, int j, int N) {
  return i*N+j;
}

void mat_mult(int N, float *A, float *B, float *C) {
  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      C[idx(i,j,N)] = 0.0f;
      for(int k=0; k<N; ++k) {
        // C(i,j) = sum_k A(i,k)*B(k,i)
        C[idx(i,j,N)] += A[idx(i,k,N)]*B[idx(k,j,N)];
      }
    }
  }
}

void load_test_case(std::vector<float>& A_in, std::vector<float>& B_in) {
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
  float C[] = {
    -3, 42, 62,
    -11, 2, 5,
    -31, 27, 54
  };

  for(int i=0; i<3*3; ++i) {
    assert(C[i] == C_in[i]);
  }
}

void run_seq_test_case() {
  const int N = 3;

  std::vector<float> h_A(N*N), h_B(N*N), h_C(N*N);
  load_test_case(h_A, h_B);
  mat_mult(N, h_A.data(), h_B.data(), h_C.data());
  check_test_result(h_C);
}

int main() {
  const int N = 3;

  //int error = setDefaultPlatform();
  //if (error < 0) return -1;

  //cl::Program program = buildProgram("vadd.cl");
  //auto vadd = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "vadd");

  run_seq_test_case();
  //cl::DeviceCommandQueue deviceQueue = cl::DeviceCommandQueue::makeDefault(
      //cl::Context::getDefault(), cl::Device::getDefault());

  // a = 1; b = 2
  //setArrayTo(h_a, 1, N);
  //setArrayTo(h_b, 2, N);

  // Create buffers and copy data
  //cl::Buffer d_a = cl::Buffer(h_a.begin(), h_a.end(), false);
  //cl::Buffer d_b = cl::Buffer(h_b.begin(), h_b.end(), false);
  //cl::Buffer d_c = cl::Buffer(CL_MEM_READ_WRITE, sizeof(int)*N);

  //// c = a+b
  //vadd(cl::EnqueueArgs(cl::NDRange(N)), d_a, d_b, d_c);

  //// b = 3
  //setArrayTo(h_b, 3, N);
  //cl::copy(h_b.begin(), h_b.end(), d_b);

  //// a = b+c = 1+2+3
  //vadd(cl::EnqueueArgs(cl::NDRange(N), cl::NDRange(WORKGROUP_SIZE)), d_c, d_b, d_a);

  //// Get result
  //cl::copy(d_a, h_a.begin(), h_a.end());

}
