#!/bin/bash

# Configuration
SRC_FILE="Kokkos_UNIT_TEST.cpp"
OUT_FILE="test.out"
REPORT_FILE="kokkos_test_report.txt"
CXX=${CXX:-g++} # Use environment variable CXX if defined, otherwise default to g++

# Set your Kokkos install path if needed
KOKKOS_DIR="$HOME/kokkos-install"  # Change this if your Kokkos is installed elsewhere

# Compiler flags (edit for CUDA if needed)
CXXFLAGS="-O3 -std=c++17 -I${KOKKOS_DIR}/include"
LDFLAGS="-L${KOKKOS_DIR}/lib -lkokkoscore -lkokkoscontainers -lkokkosalgorithms -lkokkossimd -fopenmp"

# Clean old builds
rm -f ${OUT_FILE} ${REPORT_FILE}

# Compile
echo "Compiling ${SRC_FILE}..."
$CXX ${CXXFLAGS} ${SRC_FILE} -o ${OUT_FILE} ${LDFLAGS}
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# Run and save output
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
echo "Running test..."
./${OUT_FILE} | tee ${REPORT_FILE}

