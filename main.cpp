#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#endif
#include "cl2.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <fstream>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
      // for (auto platform : platforms) {
      //    std::string name;
      //    platform.getInfo(CL_PLATFORM_NAME, &name);
      //    std::cout << name << std::endl;
      // }
      platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      // load opencl source
      std::ifstream cl_file("convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, cl_string);

      // create program
      cl::Program program(context, source);

      // compile opencl source
      program.build(devices);

      std::ifstream input_file("input.txt");
      std::ofstream output_file("output.txt");

      size_t N = 512;
      size_t M = 3;

      input_file >> N >> M;

      size_t const matrix_size = N * N;
      size_t const mask_size = M * M;
      // create a message to send to kernel

      std::vector<float> input(matrix_size, 1);
      std::vector<float> output(matrix_size, 1);
      std::vector<float> mask(mask_size, 1);
      for (size_t i = 0; i < N; ++i)
      {
         for (size_t j = 0; j < N; ++j) {
            input_file >> input[i * N + j];
         }
      }
      for (size_t i = 0; i < M; ++i)
      {
         for (size_t j = 0; j < M; ++j) {
            input_file >> mask[i * M + j];
         }
      }

      // allocate device buffer to hold message
      cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size);
      cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * matrix_size);
      cl::Buffer dev_mask  (context, CL_MEM_READ_ONLY, sizeof(float) * mask_size);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * matrix_size, &input[0]);
      queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, sizeof(float) * mask_size, &mask[0]);

      // load named kernel from opencl source
      size_t workgroup_sizeX = 16;
      size_t workgroup_sizeY = 8;
      size_t global_workgroup_sizeX = (N + workgroup_sizeX - 1) / workgroup_sizeX * workgroup_sizeX;
      size_t global_workgroup_sizeY = (N + workgroup_sizeY - 1) / workgroup_sizeY * workgroup_sizeY;
      cl::Kernel kernel_gmem(program, "gpu_convolution_gmem");
      cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int> convolution_gmem(kernel_gmem);
      cl::EnqueueArgs eargs(queue, cl::NDRange(global_workgroup_sizeX, global_workgroup_sizeY), cl::NDRange(workgroup_sizeX, workgroup_sizeY));
      cl::Event event = convolution_gmem(eargs, dev_input, dev_mask, dev_output, (int) M, (int) N);

      event.wait();
      cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end_time   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      cl_ulong elapsed_time = end_time - start_time;

      queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * matrix_size, &output[0]);
      output_file << std::setprecision(3) << std::fixed;
      for (size_t i = 0; i < N; ++i) {
         for (size_t j = 0; j < N; ++j) {
            output_file << output[i * N + j];
            if (j != N - 1) {
               output_file << " ";
            }
         }
         output_file << std::endl;
      }

   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
