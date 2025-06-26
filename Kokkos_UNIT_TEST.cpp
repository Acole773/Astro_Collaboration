#include <Kokkos_Core.hpp>
#include "Kokkos_DotProduct.hpp"
#include "Kokkos_MatrixVector.hpp"
#include <iostream>
#include <cmath>
#include <vector>

//Kokkos test file

/*****************************************/

void Initialize_Test(
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int N_g = 1000;
        bool DebugCheck = true;
        double tol = 1e-10;

	// --- dot product setup and test here ---

        // Allocate and initialize host vectors
        std::vector<double> h_Nnew(N_g, 1.0);
        std::vector<double> h_Nold(N_g, 2.0);
        std::vector<double> h_dV(N_g, 0.5);

        // Copy to device
        Kokkos::View<double*> Nnew("Nnew", N_g);
        Kokkos::View<double*> Nold("Nold", N_g);
        Kokkos::View<double*> dV("dV", N_g);

        Kokkos::deep_copy(Nnew, Kokkos::View<const double*, Kokkos::HostSpace>(h_Nnew.data(), N_g));
        Kokkos::deep_copy(Nold, Kokkos::View<const double*, Kokkos::HostSpace>(h_Nold.data(), N_g));
        Kokkos::deep_copy(dV,   Kokkos::View<const double*, Kokkos::HostSpace>(h_dV.data(), N_g));

        // Compute dot products on device
        double sum_Nnew_dV = DotProduct_Kokkos(Nnew, dV);
        double sum_Nold_dV = DotProduct_Kokkos(Nold, dV);

        if (DebugCheck) {
            // CPU-side reference computation
            double sum_Nnew_dV_ref = 0.0;
            double sum_Nold_dV_ref = 0.0;
            for (int i = 0; i < N_g; ++i) {
                sum_Nnew_dV_ref += h_Nnew[i] * h_dV[i];
                sum_Nold_dV_ref += h_Nold[i] * h_dV[i];
            }

            // Verify
            bool passed = std::abs(sum_Nnew_dV - sum_Nnew_dV_ref) < tol &&
                          std::abs(sum_Nold_dV - sum_Nold_dV_ref) < tol;

            if (passed) {
                std::cout << "DOT_SUM functioning nominally" << std::endl;
            } else {
                std::cout << "DOT_SUM ERROR" << std::endl;
                std::cout << "Device: " << sum_Nnew_dV << " " << sum_Nold_dV << std::endl;
                std::cout << "Host:   " << sum_Nnew_dV_ref << " " << sum_Nold_dV_ref << std::endl;
            }
        }

        // --- matrix-vector product test STARTS here ---
        std::vector<std::vector<double>> h_cMat(N_g, std::vector<double>(N_g));
        for (int i = 0; i < N_g; ++i)
            for (int j = 0; j < N_g; ++j)
                h_cMat[i][j] = 0.001 * (i + j + 1);

        std::vector<double> h_result(N_g, 0.0);
        std::vector<double> reference(N_g, 0.0);

        for (int i = 0; i < N_g; ++i)
            for (int j = 0; j < N_g; ++j)
                reference[i] += dt * h_cMat[i][j] * h_Nold[j];

        Kokkos::View<double**> cMat("cMat", N_g, N_g);
        for (int i = 0; i < N_g; ++i)
            for (int j = 0; j < N_g; ++j)
                cMat(i, j) = h_cMat[i][j];

        Kokkos::View<double*> Nnew_2("Nnew_2", N_g);
        Kokkos::deep_copy(Nnew_2, 0.0);

        MatrixVectorProduct_Kokkos(cMat, Nold, Nnew_2, dt);

        Kokkos::deep_copy(h_result, Nnew_2);

        bool valid = true;
        for (int i = 0; i < N_g; ++i) {
            if (std::abs(h_result[i] - reference[i]) > tol) {
                valid = false;
                break;
            }
        }

        if (valid)
            std::cout << "MATRIX_VECTOR_PRODUCT functioning nominally" << std::endl;
        else
            std::cout << "MATRIX_VECTOR_PRODUCT ERROR" << std::endl;
        // --- matrix-vector product test ENDS here ---
	}
    Kokkos::finalize();
    return 0;
}

