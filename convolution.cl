__kernel void gpu_convolution_gmem(__global float *input, __global float * mask,
                                   __global float * output, int M, int N)
{
   int idx = get_global_id(0);
   int idy = get_global_id(1);

   if (idx >= N || idy >= N) {
      return;
   }

   float res = 0;
   for (int i = 0; i < M; ++i)
   {
      for (int j = 0; j < M; ++j) {
         int input_idx = (idx + i - M / 2);
         int input_idy = (idy + j - M / 2);
         if (input_idx >= 0 && input_idx < N && input_idy >= 0 && input_idy < N)
            res += input[input_idx + input_idy * N] * mask[i + j * M];
      }
   }
   output[idx + idy * N] = res;
}