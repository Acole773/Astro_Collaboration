// Headers
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
//CUDA
#include <stdexcept>
#include <cuda_runtime.h>
#ifndef EXERCISES_INCLUDE_CUDA_CUDA_EXCEPTION_HPP
#define EXERCISES_INCLUDE_CUDA_CUDA_EXCEPTION_HPP
// Source Files
#include "Source/Logfile.h"
#include "Source/InitializeNES.h"
#include "Source/RestartCalculation.h"
#include "Source/BuildCollisionMatrix_NES.h"
#include "Source/Write_Plotfile.h"
#include "Source/ComputeRates.h"
#include "Source/NewtonRaphson.h"
#include "Source/logspace.h"
#include "Source/ApplyPerturbation.h"

using namespace std;
using namespace std::chrono;

constexpr auto default_block_size = 512;

// Wrap cuda calls to catch runtime errors
#define gpuErCh(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const std::string& file, int line) {
  if (code != cudaSuccess) {
    std::string name = cudaGetErrorName(code);
    std::string message =
        cudaGetErrorString(code);
    throw std::runtime_error{
        "Encountered CUDA error: " + name + ": " + message + " in " + file + " l." + std::to_string(line)};
  }
}


#endif //EXERCISES_INCLUDE_CUDA_CUDA_EXCEPTION_HPP

std::vector<double> flattenMatrix(const std::vector<std::vector<double>>& matrix) {
    std::vector<double> flattened;

    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }

    return flattened;
}
std::vector<std::vector<double>> reshapeMatrix(const std::vector<double>& flattened, int rows, int cols) {
    std::vector<std::vector<double>> reshaped(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            reshaped[i][j] = flattened[i * cols + j];
        }
    }

    return reshaped;
}
/*
__global__ void axpy_kernel(const int size, const double* __restrict__ alpha,  const double* __restrict__ x, double* __restrict__ y){
    const int n_threads = blockDim.x;
    const int grid_dim = gridDim.x;
    const int tidx = threadIdx.x + blockIdx.x * n_threads;

    for(int i = tidx; i < size; i += grid_dim * n_threads)
    {
        y[i] = y[i] + alpha[0] * x[i];
    }
}
*/
/*
__global__ void dot_kernel(const int size, 
			const double* __restrict__ x, 
			const double* __restrict__ y, 
			double* __restrict__ RES){
    const int n_threads = blockDim.x;
    const int tidx = threadIdx.x;
    __shared__ double TEMP[default_block_size];
    TEMP[tidx] = 0;
    for(int i = tidx; i < size; i += n_threads){
        TEMP[tidx] += x[i] * y[i];
    }
    __syncthreads();
    for (int j = n_threads / 2; j>0; j/=2){
        if (tidx < j){
            TEMP[tidx] += TEMP[tidx + j];
        }
        __syncthreads();
    }
    if(tidx == 0){
	*RES = TEMP[0];
	}

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("dot_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // Additional error handling if needed
    }

}
*/

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

double DotProduct_CUDA(const std::vector<double>& vec1, const std::vector<double>& vec2,  double* __restrict__ d_vec1, double* __restrict__ d_vec2, double* __restrict__ d_result ) {
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
}
/*
__global__ void gemv_row_parallel_kernel(const double* mat, const int num_rows,
                        const int num_cols, const double* vec,
                        const int length, double* res) {

    const int n_threads = blockDim.x;               //Save number of threads
    const int tidx = threadIdx.x;                   //Save thread index
    const int gid = tidx + blockIdx.x * n_threads;  //Calculate grid index
    if (gid <= num_rows){
       res[gid] = 0.0;                              //initialize result
       for (int j = 0; j < num_cols; j++){          //loop over number of columbs 
          res[gid] += mat[j][gid] * vec[j];   //preform matrix vector multiplication
       }
    }
}
*/
/*
__global__ void gemv_shared_memory_kernel(const double* mat, const int num_rows,
                        const int num_cols, const double* vec,
                        const int length, const double a,  double* res) {
    const int n_threads = blockDim.x;              
    const int tidx = threadIdx.x;                  
    const int bid = blockIdx.x;                      
    __shared__ double tmp[default_block_size];       
    tmp[tidx] = 0;

    for (int i = tidx; i < num_cols; i += n_threads){  
	tmp[tidx] += a * mat[bid * num_cols + i] * vec[i];
    }

    for (int i = default_block_size / 2; i>0; i /= 2){
        if (tidx < i){                                 
	    tmp[tidx] += tmp[tidx + i];
	}
	__syncthreads();                              
    }
    if (tidx == 0){
	res[bid] = tmp[0];              
}
*/
__global__ void matrixVectorProductKernel(double* cMat, double* Nold, double* Nnew_2, int N_g, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_g) {
        for (int j = 0; j < N_g; j++) {
            atomicAdd(&Nnew_2[i], dt * cMat[i * N_g + j] * Nold[j]);
        }
    }
}

void MatrixVectorProduct_CUDA(const std::vector<std::vector<double>>& cMat,
                              const std::vector<double>& Nold,
                              std::vector<double>& Nnew_2,
                              int N_g,
                              double dt, 
			      double* __restrict__ d_cMat, 
			      double* __restrict__ d_Nold, 
			      double* __restrict__ d_Nnew_2) {
    std::vector<double> flat_cMat = flattenMatrix(cMat);

    // Allocate device memory
    //double* d_cMat;
    //double* d_Nold;
    //double* d_Nnew_2;

    //gpuErCh(cudaMalloc((void**)&d_cMat, N_g * N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_Nold, N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_Nnew_2, N_g * sizeof(double)) );

    // Copy data to device
    gpuErCh(cudaMemcpy(d_cMat, &flat_cMat[0], N_g * N_g * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErCh(cudaMemcpy(d_Nold, Nold.data(), N_g * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErCh(cudaMemcpy(d_Nnew_2, Nnew_2.data(), N_g * sizeof(double), cudaMemcpyHostToDevice) );

    // Launch kernel
    int threadsPerBlock = default_block_size;
    int blocksPerGrid = (N_g + threadsPerBlock - 1) / threadsPerBlock;

    matrixVectorProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_cMat, d_Nold, d_Nnew_2, N_g, dt);
	gpuErCh(cudaDeviceSynchronize() );
    // Copy result back to host
    gpuErCh(cudaMemcpy(Nnew_2.data(), d_Nnew_2, N_g * sizeof(double), cudaMemcpyDeviceToHost) );

    // Free device memory
    //gpuErCh(cudaFree(d_cMat) );
    //gpuErCh(cudaFree(d_Nold) );
    //gpuErCh(cudaFree(d_Nnew_2) );
}
/*
void gemv_shared_memory(const ValueType* mat, const size_type num_rows,
                        const size_type num_cols, const ValueType* vec,
                        const size_type length, ValueType* res) {
     gemv_shared_memory_kernel<<<dim3(num_rows), dim3(default_block_size)>>>(mat,num_rows,num_cols,vec,length,res);
}

*/	

__global__ void buildCollisionMatrixKernel(
    double* A, double* k, double* R_In, double* R_Out,
    double* N, double* dV, int N_g, double tol
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_g) {
        for (int j = 0; j < N_g; j++) {
            double N_Eq_i = 0.0;
            double N_Eq_j = 0.0;

            if (j != i) {
                double B = (R_In[i * N_g + j] * dV[i] + R_Out[i * N_g + j] * dV[j]);
                double C = dV[i] * N[i] + dV[j] * N[j];
                double a = (R_In[i * N_g + j] - R_Out[i * N_g + j]) * dV[i];
                double b = B + (R_In[i * N_g + j] - R_Out[i * N_g + j]) * C;
                double c = R_In[i * N_g + j] * C;
                double d = b * b - 4.0 * a * c;
                N_Eq_i = 0.5 * (b - sqrt(d)) / a;
                N_Eq_j = (C - (N_Eq_i * dV[i])) / dV[j];
            } else {
                N_Eq_i = N[i];
                N_Eq_j = N[j];
            }

            double diff_i = abs(N_Eq_i - N[i]) / fmax(1e-16, N_Eq_i);
            double diff_j = abs(N_Eq_j - N[j]) / fmax(1e-16, N_Eq_j);

            if (diff_i > tol || diff_j > tol) {
                A[i * N_g + j] += (1.0 - N[i]) * R_In[i * N_g + j] * dV[j];
                A[i * N_g + i] -= R_Out[i * N_g + j] * dV[j] * (1.0 - N[j]);
                k[i] += R_Out[i * N_g + j] * dV[j] + (R_In[i * N_g + j] * dV[j] - R_Out[i * N_g + j] * dV[j]) * N[j];
            }
        }
    }
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> BuildCollisionMatrix_NES_CUDA(
    std::vector<std::vector<double>>& R_In,
    std::vector<std::vector<double>>& R_Out,
    std::vector<double>& N,
    std::vector<double>& dV,
    int N_g,
    double tol,
    double* __restrict__ d_R_In,
    double* __restrict__ d_R_Out,
    double* __restrict__ d_N,
    double* __restrict__ d_dV,
    double* __restrict__ d_A,
    double* __restrict__ d_k
) {
    std::pair<std::vector<std::vector<double>>, std::vector<double>> result;

    std::vector<std::vector<double>>& A = result.first;
    std::vector<double>& k = result.second;

    A.resize(N_g);
    for (int i = 0; i < N_g; i++) {
        A[i].resize(N_g, 0.0);
    }

    k.resize(N_g, 0.0);

    // Allocate device memory
    
    //double* d_R_In;
    //double* d_R_Out;
    //double* d_N;
    //double* d_dV;
    //double* d_A;
    //double* d_k;
    std::vector<double> flat_R_In = flattenMatrix(R_In);
    std::vector<double> flat_R_Out = flattenMatrix(R_Out);
    std::vector<double> flat_A = flattenMatrix(A);
    
    //gpuErCh(cudaMalloc((void**)&d_R_In, N_g * N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_R_Out, N_g * N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_N, N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_dV, N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_A, N_g * N_g * sizeof(double)) );
    //gpuErCh(cudaMalloc((void**)&d_k, N_g * sizeof(double)) );

    // Copy data to device
    gpuErCh(cudaMemcpy(d_R_In, &flat_R_In[0], N_g * N_g * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErCh(cudaMemcpy(d_R_Out, &flat_R_Out[0], N_g * N_g * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErCh(cudaMemcpy(d_N, &N[0], N_g * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErCh(cudaMemcpy(d_dV, &dV[0], N_g * sizeof(double), cudaMemcpyHostToDevice) );

    // Launch kernel
    int threadsPerBlock = default_block_size;
    int blocksPerGrid = (N_g + threadsPerBlock - 1) / threadsPerBlock;

    buildCollisionMatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_k, d_R_In, d_R_Out, d_N, d_dV, N_g, tol
    );
	gpuErCh(cudaDeviceSynchronize() );

    // Copy results back to host
    gpuErCh(cudaMemcpy(&flat_A[0], d_A, N_g * N_g * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErCh(cudaMemcpy(&k[0], d_k, N_g * sizeof(double), cudaMemcpyDeviceToHost) );

    A = reshapeMatrix(flat_A, N_g, N_g);
    // Free device memory
    //gpuErCh(cudaFree(d_R_In) );
    //gpuErCh(cudaFree(d_R_Out) );
    //gpuErCh(cudaFree(d_N) );
    //gpuErCh(cudaFree(d_dV) );
    //gpuErCh(cudaFree(d_A) );
    //gpuErCh(cudaFree(d_k) );

    return result;
}

/*
void MemoryAllocation(){

    double* d_vec1;
    double* d_vec2;
    double* d_result;

    gpuErCh(cudaMalloc((void**)&d_vec1, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_vec2, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_result, sizeof(double)) );


    double* d_cMat;
    double* d_Nold;
    double* d_Nnew_2;

    gpuErCh(cudaMalloc((void**)&d_cMat, N_g * N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_Nold, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_Nnew_2, N_g * sizeof(double)) );


    double* d_R_In;
    double* d_R_Out;
    double* d_N;
    double* d_dV;
    double* d_A;
    double* d_k;

    gpuErCh(cudaMalloc((void**)&d_R_In, N_g * N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_R_Out, N_g * N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_N, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_dV, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_A, N_g * N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_k, N_g * sizeof(double)) );

}
void MemoryDeAllocation(){

    gpuErCh(cudaFree(d_vec1) );
    gpuErCh(cudaFree(d_vec2) );
    gpuErCh(cudaFree(d_result) );


    gpuErCh(cudaFree(d_cMat) );
    gpuErCh(cudaFree(d_Nold) );
    gpuErCh(cudaFree(d_Nnew_2) );


    gpuErCh(cudaFree(d_R_In) );
    gpuErCh(cudaFree(d_R_Out) );
    gpuErCh(cudaFree(d_N) );
    gpuErCh(cudaFree(d_dV) );
    gpuErCh(cudaFree(d_A) );
    gpuErCh(cudaFree(d_k) );

}
*/

int main() {
    
    // --- Computational Parameters ----------------------------
    string Model = "003"; // --- Determines Local Thermodynamic Conditions
    int N_g = 40;    // --- Number of Energy Groups (Must be 40)
    double t_end = 1.0e-02; // [ s ] Final time
    double t = 1.0e-15; // [ s ] Initial time
    double t_W0 = 1.0e-11; // [ s ] Initial write time
    double dt_min = 1.0e-16; // [ s ] Minimum time step
    double dt_max = 0.10; // [ s ] Maximum time step
    double dt = 1.0e-15; // [ s ] Initial time step
    double dt_grw = 1.03;    // Max dt growth per time step
    double dt_dec = 0.90;    // Decline factor for dt if restep
    double dt_FE = dt, dt_EA = dt, dt_PE = dt;
    int reStart = 0;       // For restarting run at a specific data file
    double G_A = 7.5e-01;    // Gaussian Amplitude
    double G_B = 1.0e+02;    // Gaussian Expected Value
    double G_C = sqrt(50.0); // Gaussian Width
    int cycleM = (int)1e9;    // Maximum Cycles
    int cycleD = 100;     // Display Interval
    int cycleW = 10;      // Write Interval
    double tolPE = 1.0e-01;  // Partial Equilibrium Tolerance
    double tolC = 1.0e-06;  // Particle Conservation Tolerance
    double tolBE = 1.0e-00;  // Convergence Tolerance For Backward Euler
    double tolN = 1.0e-02;  // Relative Density Tolerance For Methods 
    int FE    = 0;
    int EA    = 1;
    int FE_PE = 2;
    int EA_C  = 3;
    int QSS1  = 4;
    int QSS2  = 5;
    int BE    = 6;
    int FP    = 7;
    int QSS3  = 8;
    int AFP   = 9;
    string Scheme = "ExplicitAsymptotic";
    string Comment = "";
    string PlotFileDir = "./Output";
    string PlotFileName = "PlotFile";
    int PlotFileNumber = 0;
    string RestartDir = "./Output";
    string RestartFileName = "PlotFile";
    int RestartFileNumber = 0;
    int nPlotFiles = 175; 
    int PertCase = 2;
    double amp = 0.0;
    bool AppPert = false;
//**************************************//
    bool DebugCheck = true;
    bool Time_Crono = true;

//**************************************//
/*
    double *MAT_c;
    double *MAT_res_c;
    double *VEC1_c;
    double *VEC2_c;
    double *Nres_c;
    double res[1];
    res[0] = 0;
    double *res_c;

    cudaMalloc((void**)&MAT_c,N_g*N_g * sizeof(double));
    cudaMalloc((void**)&VEC1_c, N_g * sizeof(double));
    cudaMalloc((void**)&VEC2_c, N_g * sizeof(double));
    cudaMalloc((void**)&res_c, sizeof(res));
    //cudaMalloc((void**)&res_c, 1 * sizeof(double));
    cudaMalloc((void**)&MAT_res_c,N_g*N_g * sizeof(double));
    cudaMalloc((void**)&Nres_c, N_g * sizeof(double));
*/
    //vector<double> mat_flat(N_g*N_g);
//**************************************//

    double* d_vec1;
    double* d_vec2;
    double* d_result;

    gpuErCh(cudaMalloc((void**)&d_vec1, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_vec2, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_result, sizeof(double)) );


    double* d_cMat;
    double* d_Nold;
    double* d_Nnew_2;

    gpuErCh(cudaMalloc((void**)&d_cMat, N_g * N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_Nold, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_Nnew_2, N_g * sizeof(double)) );


    double* d_R_In;
    double* d_R_Out;
    double* d_N;
    double* d_dV;
    double* d_A;
    double* d_k;

    gpuErCh(cudaMalloc((void**)&d_R_In, N_g * N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_R_Out, N_g * N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_N, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_dV, N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_A, N_g * N_g * sizeof(double)) );
    gpuErCh(cudaMalloc((void**)&d_k, N_g * sizeof(double)) );

    // --- Logs Important Parameters Into A File ---
    Logfile(t_end, t, t_W0, dt, G_A, G_B, G_C, tolC, Scheme, Comment);
 

                  
    std::vector<double> eC(N_g), dV(N_g), N_Eq(N_g);
    std::vector<std::vector<double> > R_In(N_g), R_Out(N_g);

    InitializeNES(Model, N_g, eC, dV, R_In, R_Out, N_Eq);

    vector<double> N_0(N_g);

    if (reStart == 1) {
        N_0 = RestartCalculation(RestartDir, RestartFileName, RestartFileNumber);
    } else {
        for (int i = 0; i < N_g; i++) {
            N_0[i] = G_A * exp(-0.5 * pow((eC[i] - G_B) / G_C, 2));
        }
    }
    vector<double> Nold = N_0;
    vector<vector<double> > cMat(N_g);
    vector<double> kVec(N_g);

    auto Cmat_K0 = BuildCollisionMatrix_NES_CUDA(R_In, R_Out, Nold, dV, N_g, tolC, d_R_In, d_R_Out, d_N, d_dV, d_A, d_k);
    cMat = Cmat_K0.first;
    kVec = Cmat_K0.second;
    if (DebugCheck == true){
        auto Cmat_K0_2 = BuildCollisionMatrix_NES(R_In, R_Out, Nold, dV, N_g, 0.0, cMat, kVec);
        auto cMat_2 = Cmat_K0_2.first;
        auto kVec_2 = Cmat_K0_2.second;
         if( cMat == cMat_2 && kVec == kVec_2){
            cout << "BuildCollisionMatrix functioning Nominaly" << endl;
        }
            else{ cout << "BuildCollisionMatrix ERROR" <<endl; }
    }
    bool done = false;
    bool reStep = false;
    int cycle = 0;
    int true_cycle = 0;
    int nIterations = 0;
    int nTrueIterations = 0;
    int maxFPIterations = 10000; 
    int mAA = 3;
    // --- Write Initial Condition ---
    PlotFileNumber = Write_Plotfile(t, dt, Nold, eC, dV, kVec, cMat, 0, 0, 0, 0, FE, dt, dt, dt, tolC, dt_grw, dt_max,
                                    PlotFileDir, PlotFileName, PlotFileNumber);
    std::vector<double> wrtTimes = logspace(log10(t_W0), log10(t_end), nPlotFiles);
    int wrtCount = 1;

    
	double Time_Sum_1 = 0.0;
	int Time_idx_1 = 0;
	double Time_Sum_2 = 0.0;
	int Time_idx_2 = 0;
   

        // --- Main Time Loop ---
    while (!done) {


        true_cycle++;

            if (!reStep)
        {
            cycle++;
           // std::cout<<"Current cycle: " << cycle << std::endl; // Print out current cycle
        }

        // --- Check for Maximum Cycles ---
        if (true_cycle >= cycleM) {
            cout << "Maximum Number of Cycles Exceeded. Exiting Program." << endl;
            done = true;
            break;
        }

        // ---- Main Loop ----
    if (AppPert == true){
        Nold = ApplyPerturbation(Nold, amp, dV, N_g, PertCase);
        }

    
        //initialize Nold and Nnew with appropriate values
        std::vector<double> Nnew(N_g);

        if (Scheme == "ExplicitAsymptotic") {
           // int Branch = EA
	    	auto start_1 = high_resolution_clock::now();
            auto Cmat_K0 = BuildCollisionMatrix_NES_CUDA(R_In, R_Out, Nold, dV, N_g, tolC, d_R_In, d_R_Out, d_N, d_dV, d_A, d_k);
            	auto stop_1 = high_resolution_clock::now();
                auto duration_1 = duration_cast<microseconds>(stop_1 - start_1);
		Time_Sum_1 += std::chrono::duration<double>(duration_1).count();
		Time_idx_1 ++;
	    cMat = Cmat_K0.first;
            kVec = Cmat_K0.second;
	if (DebugCheck == true){
                auto start_2 = high_resolution_clock::now();
	    auto Cmat_K0_2 = BuildCollisionMatrix_NES(R_In, R_Out, Nold, dV, N_g, 0.0, cMat, kVec);
                auto stop_2 = high_resolution_clock::now();
                auto duration_2 = duration_cast<microseconds>(stop_2 - start_2);
                Time_Sum_2 += std::chrono::duration<double>(duration_2).count();
                Time_idx_2 ++;
	    auto cMat_2 = Cmat_K0_2.first;
            auto kVec_2 = Cmat_K0_2.second;
	    if( cMat == cMat_2 && kVec == kVec_2){
		cout << "BuildCollisionMatrix functioning Nominaly" << endl;
	    }
	    else{ cout << "BuildCollisionMatrix ERROR" <<endl; }
	}

            double dt_FE = 1.0 / kVec[0];
                for (int i = 1; i < kVec.size(); i++) {
                double dt_i = 1.0 / kVec[i];
                if (dt_i < dt_FE) {
                    dt_FE = dt_i;
                    }
                }
            if (reStep) {
                dt = dt_dec * dt;
                reStep = false;
            }
            else {
                dt = dt_grw * dt;
                    if (dt > dt_max * t){
                        dt = dt_max * t;
                    }
            }

            if (dt <= dt_FE) {
                Nnew = Nold;
		/*
		mat_flat = flattenMatrix(cMat);
		cudaMemcpy(MAT_c, mat_flat.data(), mat_flat.size() * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(VEC1_c, Nold.data(), Nold.size()*sizeof(double), cudaMemcpyHostToDevice);
		gemv_shared_memory_kernel<<<dim3(N_g), dim3(default_block_size)>>>(MAT_c, N_g, N_g, VEC1_c, N_g, dt, Nres_c);
		cudaDeviceSynchronize(); 
		cudaMemcpy(Nnew.data(), Nres_c, Nnew.size() * sizeof(double), cudaMemcpyDeviceToHost);
		*/
		MatrixVectorProduct_CUDA(cMat, Nold, Nnew, N_g, dt, d_cMat, d_Nold, d_Nnew_2);
		if (DebugCheck == true){
		    std::vector<double> Nnew_2 = Nold;
            	    for (int i = 0; i < N_g; i++) {
                        for (int j = 0; j < N_g; j++) {
                           Nnew_2[i] += dt * cMat[i][j] * Nold[j];
                        }
                    }
		    if( Nnew == Nnew_2 ){
                	cout << "POP_CAL functioning Nominaly" << endl;
            	    }
            		else{ 
			    cout << "POP_CAL ERROR" <<endl; 
			}
		}
	    }
            else {
                Nnew = Nold;
                for (int i = 0; i < N_g; i++) {
                    for (int j = 0; j < N_g; j++) {
                        Nnew[i] += (dt / (1 + kVec[i] * dt)) * cMat[i][j] * Nold[j];
                    }
                }   
                //double sum_Nnew_dV = 0, sum_Nold_dV = 0;
		/*
		cudaMemcpy(VEC1_c, Nnew.data(), Nnew.size()*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(VEC2_c, dV.data(), dV.size()*sizeof(double), cudaMemcpyHostToDevice);
		dot_kernel<<<dim3(1), dim3(default_block_size)>>>(N_g, VEC1_c, VEC2_c, res_c);
		cudaMemcpy(res, res_c, sizeof(double), cudaMemcpyDeviceToHost);
		sum_Nnew_dV = res[0];

                cudaMemcpy(VEC1_c, Nold.data(), Nnew.size()*sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(VEC2_c, dV.data(), dV.size()*sizeof(double), cudaMemcpyHostToDevice);
                dot_kernel<<<dim3(1), dim3(default_block_size)>>>(N_g, VEC1_c, VEC2_c, res_c);
                cudaMemcpy(res, res_c, sizeof(double), cudaMemcpyDeviceToHost);
                sum_Nold_dV = res[0];
		*/
		double sum_Nnew_dV = DotProduct_CUDA(Nnew, dV, d_vec1, d_vec2, d_result);
    		double sum_Nold_dV = DotProduct_CUDA(Nold, dV, d_vec1, d_vec2, d_result);
		if (DebugCheck == true){
		    double sum_Nnew_dV_2 = 0, sum_Nold_dV_2 = 0;
                    for (int i = 0; i < N_g; i++) {
                        sum_Nnew_dV_2 += Nnew[i] * dV[i];
                        sum_Nold_dV_2 += Nold[i] * dV[i];
                    }
		    if( abs(sum_Nnew_dV-sum_Nnew_dV_2) < 1e-10 && abs(sum_Nold_dV-sum_Nold_dV_2)<1e-10){
                        cout << "DOT_SUM functioning Nominaly" << endl;
                    }
                        else{ cout << "DOT_SUM ERROR" <<endl; }

		}
		
            if (abs(sum_Nnew_dV - sum_Nold_dV) / sum_Nold_dV > tolC) {
                reStep = true;
            }
            double max_rel_diff = 0;
            for (int i = 0; i < N_g; i++) {
                max_rel_diff = max(max_rel_diff, abs(Nnew[i] - Nold[i]) / max(Nold[i], 1e-8));
            }
            if (max_rel_diff > tolN) {
                reStep = true;
            }
        }
    }

        
        else if (Scheme == "FE_PE") {
            int Branch = FE_PE;
            auto Cmat_K0 = BuildCollisionMatrix_NES(R_In, R_Out, Nold, dV, N_g, tolPE, cMat, kVec);
            cMat = Cmat_K0.first;
            kVec = Cmat_K0.second;
            double dt_PE = *std::min_element(kVec.begin(), kVec.end());
            dt = min(dt_PE, dt_grw * dt);
            Nnew = Nold;
            for (int i = 0; i < N_g; i++) {
                for (int j = 0; j < N_g; j++) {
                    Nnew[i] += dt * cMat[i][j] * Nold[j];
        		}
    		}
	}

        
        else if (Scheme == "QSS1") {
            //Initalize exp_kdt
            std::vector<double> exp_kdt(N_g,0);
            std::vector<double> inv_k(N_g,0);
            int Branch = QSS1;

            if( reStep )
            {
                dt = dt_dec * dt;
                reStep = false;
            }
            else
            {
                dt = dt_grw * dt;
            }
            auto F0_k0 = ComputeRates(R_In, R_Out, Nold, dV);
            std::vector<double> &F0 = F0_k0.first;
            std::vector<double> &k0 = F0_k0.second;
            kVec = k0;

            for(int i = 0; i < N_g; i++) {
                exp_kdt[i] = exp( - dt * k0[i] );
                inv_k[i] = 1.0 / k0[i];
            }
            
            dt_FE = *std::min_element(inv_k.begin(), inv_k.end());
            

            if( dt <= dt_FE ){
                auto Cmat_K0 = BuildCollisionMatrix_NES( R_In, R_Out, Nold, dV, N_g, 0.0, cMat, kVec );
                cMat = Cmat_K0.first;
                kVec = Cmat_K0.second;
       
                std::vector<double> Nnew_temp(N_g);    
                Nnew_temp = Nold;
                for (int i = 0; i < N_g; i++) {
                    for (int j = 0; j < N_g; j++) {
                        Nnew_temp[i] += dt * cMat[i][j] * Nold[j];  //ForwardEulerupdate
                        }
                    }
                
                Nnew = Nnew_temp; // Replace the original Nnew with the updated values
                nTrueIterations++;
            }
            else
            {
                auto Cmat_K0 = BuildCollisionMatrix_NES( R_In, R_Out, Nold, dV, N_g, 0.0, cMat, kVec );
                cMat = Cmat_K0.first;
                kVec = Cmat_K0.second;
                std::vector<double> Nnew_temp(N_g);
                Nnew_temp = Nold;
                std::vector<double> MM_temp(N_g);

                for(int i = 0; i < N_g; i++) {
                    for(int j = 0; j < N_g; j++) {
                        MM_temp[i] = ( cMat[i][j] * Nold[i] );
                    }
                }
                for(int i = 0; i < N_g; i++) {
                    Nnew_temp[i] +=  MM_temp[i] * ( 1.0 - exp_kdt[i] ) / k0[i]; //qss1 update
                }
                Nnew = Nnew_temp; // Replace the original Nnew with the updated values
            }
            double sum_Nnew_dV = std::inner_product(Nnew.begin(), Nnew.end(), dV.begin(), 0.0);
            double sum_Nold_dV = std::inner_product(Nold.begin(), Nold.end(), dV.begin(), 0.0);

            if (std::abs((sum_Nnew_dV - sum_Nold_dV) / sum_Nold_dV) > tolC) {
                reStep = true; }
        }
    
        else if (Scheme == "QSS2") {

            //Initalize exp_kdt
            std::vector<double> exp_kdt(N_g,0);
            std::vector<double> inv_k(N_g,0);
            int Branch = QSS2;

            if( reStep )
            {
                dt = dt_dec * dt;
                reStep = false;
            }
            else
            {
                dt = dt_grw * dt;
            }
            auto F0_k0 = ComputeRates(R_In, R_Out, Nold, dV);
            std::vector<double> &F0 = F0_k0.first;
            std::vector<double> &k0 = F0_k0.second;
            kVec = k0;

            for(int i = 0; i < N_g; i++) {
                exp_kdt[i] = exp( - dt * k0[i] );
                inv_k[i] = 1.0 / k0[i];
            }
            
            dt_FE = *std::min_element(inv_k.begin(), inv_k.end());
            

            if( dt <= dt_FE ){
                auto Cmat_K0 = BuildCollisionMatrix_NES( R_In, R_Out, Nold, dV, N_g, 0.0, cMat, kVec );
                cMat = Cmat_K0.first;
                kVec = Cmat_K0.second;
       
                std::vector<double> Nnew_temp(N_g);    
                Nnew_temp = Nold;
                for (int i = 0; i < N_g; i++) {
                    for (int j = 0; j < N_g; j++) {
                        Nnew_temp[i] += dt * cMat[i][j] * Nold[j];  //ForwardEulerupdate
                        }
                    }
                
                Nnew = Nnew_temp; // Replace the original Nnew with the updated values
                nTrueIterations++;
            }
            else 
            {

                //***************************//
                std::vector<double> r0(N_g);    //r0
                std::vector<double> Alpha0(N_g); //The First Alpha
                std::vector<double> Np(N_g);  // Np
                //                  kp = Fp_kp.second;
                //                  Fp = Fp_kp.first;
                std::vector<double> rBAR(N_g);
                std::vector<double> AlphaBAR(N_g); //Corrected Alpha
                std::vector<double> Ft(N_g);
                //std::vector<double> Nnew(N_g);

                //***************************//
                
                //r0 = exp_kdt;
                for (int i = 0; i < N_g; i++) {
                    r0[i] = 1.0 / (k0[i] * dt);
                }   
         
                
                for (int i = 0; i < N_g; i++) {
                Alpha0[i] = ((160 * pow(r0[i],3)) + (60 * pow(r0[i],2)) + (11 * r0[i]) + 1)/
                            ((360 * pow(r0[i],3)) + (60 * pow(r0[i],2)) + (12 * r0[i]) + 1);
                } 

                Np = Nold;
                for (int i = 0; i < N_g; i++) {
                    Np[i] += dt * (F0[i] - (k0[i] * Nold[i])) / (1.0 + (Alpha0[i] * k0[i] * dt));
                }
         
                auto Fp_kp = ComputeRates( R_In, R_Out, Np, dV );
                std::vector<double> kp = Fp_kp.second;
                                    Ft = Fp_kp.first;

                Nnew = Nold;

                std::vector<double> kBAR(N_g);
                for (int i = 0; i < N_g; i++) {
                    kBAR[i] = 0.5 * (kp[i] + k0[i]);
                }
                for (int i = 0; i < N_g; i++) {
                    rBAR[i] = 1.0 / (kBAR[i] * dt);
                }
                for (int i = 0; i < N_g; i++) {
                    AlphaBAR[i] = ((160 * pow(rBAR[i],3)) + (60 * pow(rBAR[i],2)) + (11 * rBAR[i]) + 1)/
                                  ((360 * pow(rBAR[i],3)) + (60 * pow(rBAR[i],2)) + (12 * rBAR[i]) + 1);
                }
                for (int i = 0; i < N_g; i++){
                    Ft[i] = AlphaBAR[i] * Ft[i] + (1.0 - AlphaBAR[i]) * F0[i];
                }
                for (int i = 0; i < N_g; i++){
                    Nnew[i] += dt * (Ft[i] - kBAR[i] * Nold[i]) / (1.0 + (Alpha0[i] * kBAR[i] * dt));
                }
            nTrueIterations += 2;
            }
            double sum_Nnew_dV = std::inner_product(Nnew.begin(), Nnew.end(), dV.begin(), 0.0);
            double sum_Nold_dV = std::inner_product(Nold.begin(), Nold.end(), dV.begin(), 0.0);

            if (std::abs((sum_Nnew_dV - sum_Nold_dV) / sum_Nold_dV) > tolC) {
                reStep = true; 
            }
        }
    
        else {
        int Branch = FE;
        auto Cmat_K0 = BuildCollisionMatrix_NES(R_In, R_Out, Nold, dV, N_g, 0.0, cMat, kVec);
        cMat = Cmat_K0.first;
        kVec = Cmat_K0.second;

        dt = std::min(*std::min_element(kVec.begin(), kVec.end()), dt_grw * dt);

        std::vector<double> Nnew(N_g);
        for (int i = 0; i < N_g; i++) {
            Nnew[i] = Nold[i] + dt * std::inner_product(cMat[i].begin(), cMat[i].end(), Nold.begin(), 0.0);
        }  

        if (!reStep) {
            Nold = Nnew;
            t += dt;
        }
        }


    // End of case switch

        // --- Display Current Time ---
        if (cycle % cycleD == 0) {
             cout << "Time since simulation initiation: " << t << " seconds passed" << endl;
             cout << "dt: " << dt << endl;

        }
        
        if ( t >= wrtTimes[wrtCount]) {
        // --- Write Plotfile ---
        //if ((t - t_W0) >= 0.0 && (cycle % cycleW == 0)) 
            PlotFileNumber = Write_Plotfile(t, dt, Nold, eC, dV, kVec, cMat, 0, 0, 0, 0, FE, dt, dt, dt, tolC, dt_grw, dt_max,
                                    PlotFileDir, PlotFileName, PlotFileNumber);
            wrtCount = wrtCount +1;
        
        }
       
        // --- Time Step Update---
        if (!reStep) {
            
            Nold = Nnew;
            t += dt;
        }

        if ((t - t_end) >= 0) {
            done = true;

            //break;
        }
    }
cout << "Time since simulation initiation: " << t << " seconds passed" << endl;
cout << "dt: " << dt << endl;

    gpuErCh(cudaFree(d_vec1) );
    gpuErCh(cudaFree(d_vec2) );
    gpuErCh(cudaFree(d_result) );


    gpuErCh(cudaFree(d_cMat) );
    gpuErCh(cudaFree(d_Nold) );
    gpuErCh(cudaFree(d_Nnew_2) );


    gpuErCh(cudaFree(d_R_In) );
    gpuErCh(cudaFree(d_R_Out) );
    gpuErCh(cudaFree(d_N) );
    gpuErCh(cudaFree(d_dV) );
    gpuErCh(cudaFree(d_A) );
    gpuErCh(cudaFree(d_k) );


if (Time_Crono == true){
double avg_t_1 = Time_Sum_1/Time_idx_1;
double avg_t_2 = Time_Sum_2/Time_idx_2;

printf("The average time of GPU_BCM is %f seconds\n", avg_t_1);
printf("The average time of CPU_BCM is %f seconds\n", avg_t_2);
}
/*
cudaFree(Nres_c);
cudaFree(res_c);
cudaFree(VEC2_c);
cudaFree(VEC1_c);
cudaFree(MAT_res_c);
cudaFree(MAT_c);
*/
return 0;
}
//}  // namespace cuda

