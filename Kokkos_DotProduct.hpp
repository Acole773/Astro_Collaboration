#pragma once
#include <Kokkos_Core.hpp>
#include <vector>
#include <iostream>
#include <Kokkos_Atomic.hpp>

// This functor computes the dot product between two 1D Views of doubles.
struct DotFunctor {
    Kokkos::View<const double*> x;
    Kokkos::View<const double*> y;

    DotFunctor(Kokkos::View<const double*> x_, Kokkos::View<const double*> y_)
        : x(x_), y(y_) {}

    // Each thread computes a partial product for a single index.
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& sum) const {
        sum += x(i) * y(i);
    }
};

// This function performs a dot product using Kokkos parallel_reduce.
// It assumes the input views are device-accessible and of equal size.
double DotProduct_Kokkos(Kokkos::View<double*> vec1, Kokkos::View<double*> vec2) {
    int N_g = vec1.extent(0);
    double result = 0.0;

    if (vec2.extent(0) != N_g) {
        std::cerr << "Error: Input vectors must be the same size.\n";
        return NAN;
    }

    // Perform the dot product with a reduction over all elements.
    Kokkos::parallel_reduce("dot_product",
                            Kokkos::RangePolicy<>(0, N_g),
                            DotFunctor(vec1, vec2),
                            result);

    return result;
}

