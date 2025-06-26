//Kokkos test file

using Kokkos::atomic_add;
using Kokkos::PerTeam;
using Kokkos::Sum;
using Kokkos::TeamPolicy;
using Kokkos::parallel_for;

struct DotFunctor {
    Kokkos::View<const double*> x;
    Kokkos::View<const double*> y;

    DotFunctor(Kokkos::View<const double*> x_, Kokkos::View<const double*> y_)
        : x(x_), y(y_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& sum) const {
        sum += x(i) * y(i);
    }
};

double DotProduct_Kokkos(Kokkos::View<double*> vec1, Kokkos::View<double*> vec2) {
    int N_g = vec1.extent(0);
    double result = 0.0;

    // Device view to store the result
    Kokkos::View<double> d_result("d_result");

    // Launch parallel reduction
    Kokkos::parallel_reduce("dot_product", Kokkos::RangePolicy<>(0, N_g), DotFunctor(vec1, vec2), result);

    return result;
}
/*
Explanation of the Changes:

    Using Kokkos::View Properly: Instead of manually copying to device memory, we use Kokkos::View for vec1 and vec2, which can be allocated on the device.

    Functor for parallel_reduce: Kokkos prefers functors or lambda functions for parallel_reduce. This functor performs the element-wise multiplication and accumulates the result.

    Avoiding Shared Memory Management: Instead of manually reducing in shared memory, Kokkos handles reductions efficiently behind the scenes with parallel_reduce.

*/


/*
__global__ void dot_kernel(const int size, const double* __restrict__ x,
                           const double* __restrict__ y, double* __restrict__ RES) {
    const int n_threads = blockDim.x;
    const int tidx = threadIdx.x;
    extern __shared__ double TEMP[];

    TEMP[tidx] = 0;

    for (int i = tidx; i < size; i += n_threads) {
        TEMP[tidx] += x[i] * y[i];
    }

    __syncthreads();

    for (int j = n_threads / 2; j > 0; j /= 2) {
        if (tidx < j) {
            TEMP[tidx] += TEMP[tidx + j];
        }
        __syncthreads();
    }

    if (tidx == 0) {
        *RES = TEMP[0];
    }
}
double DotProduct_Kokkos(Kokkos::View<double> vec1, Kokkos::View<double> vec2,  double* __restrict__ d_vec1, double* __restrict__ d_vec2, double* __restrict__ d_result ) {
    int N_g = vec1.size();
    double result = 0.0;

    // Allocate device memory
    //double* d_vec1;
    //double* d_vec2;
    //double* d_result;

    //gpuErCh(cudaMalloc((void**)&d_vec1, N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_vec2, N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_result, sizeof(double)) );

    // Copy data to device
    gpuErCh(cudaMemcpy(d_vec1, vec1.data(), N_g * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErCh(cudaMemcpy(d_vec2, vec2.data(), N_g * sizeof(double), cudaMemcpyHostToDevice) );

    // Launch kernel
    int threadsPerBlock = default_block_size;
    int blocksPerGrid = (N_g + threadsPerBlock - 1) / threadsPerBlock;

    dot_kernel<<<blocksPerGrid, threadsPerBlock, default_block_size*sizeof(double)>>>(N_g, d_vec1, d_vec2, d_result);
    gpuErCh(cudaDeviceSynchronize() );

    // Copy result back to host
    gpuErCh(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost) );

    // Free device memory
    //gpuErCh(cudaFree(d_vec1) );
    //gpuErCh(cudaFree(d_vec2) );
    //gpuErCh(cudaFree(d_result) );

    return result;
}*/
