// Kokkos_MatrixVector.hpp
#pragma once
#include <Kokkos_Core.hpp>

// Optional fast version (no atomics required if each i is unique)
struct MatrixVectorProductFunctor {
    Kokkos::View<const double**> cMat;
    Kokkos::View<const double*> Nold;
    Kokkos::View<double*> Nnew_2;
    double dt;

    MatrixVectorProductFunctor(Kokkos::View<const double**> cMat_,
                               Kokkos::View<const double*> Nold_,
                               Kokkos::View<double*> Nnew_2_,
                               double dt_)
        : cMat(cMat_), Nold(Nold_), Nnew_2(Nnew_2_), dt(dt_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        double sum = 0;
        for (int j = 0; j < Nold.extent(0); ++j) {
            sum += dt * cMat(i, j) * Nold(j);
        }
        Nnew_2(i) = sum;
    }
};

// Entry point for calling from host
void MatrixVectorProduct_Kokkos(Kokkos::View<double**> cMat,
                                Kokkos::View<double*> Nold,
                                Kokkos::View<double*> Nnew_2,
                                double dt) {
    int N_g = Nold.extent(0);
    Kokkos::parallel_for("matrix_vector_product",
                         Kokkos::RangePolicy<>(0, N_g),
                         MatrixVectorProductFunctor(cMat, Nold, Nnew_2, dt));
}

