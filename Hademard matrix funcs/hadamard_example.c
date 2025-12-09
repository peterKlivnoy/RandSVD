/*
 * Simple Fast Walsh-Hadamard Transform Implementation
 * This is a basic example - optimized versions would use SIMD, threading, etc.
 */

#include <stdlib.h>
#include <math.h>

/* 
 * In-place Fast Walsh-Hadamard Transform
 * Input: data array of size n (must be power of 2)
 * Output: transformed data (in-place)
 */
void fwht_1d(double *data, int n) {
    int h = 1;
    while (h < n) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                double x = data[j];
                double y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/*
 * 2D Fast Walsh-Hadamard Transform (row-wise)
 * Applies FWHT to each row of a matrix stored in row-major order
 * 
 * Parameters:
 *   data: pointer to matrix data (row-major: data[i*n + j] = matrix[i][j])
 *   m: number of rows
 *   n: number of columns (must be power of 2)
 */
void fwhtKernel2dOrdinary(double *data, int m, int n) {
    for (int i = 0; i < m; i++) {
        fwht_1d(data + i * n, n);
    }
}

/*
 * Python-friendly wrapper that allocates output array
 * This would be called from Python via SWIG
 */
void fwhtKernel2dOrdinaryWrapper(double *input, int m, int n, double *output) {
    // Copy input to output
    for (int i = 0; i < m * n; i++) {
        output[i] = input[i];
    }
    // Transform in-place
    fwhtKernel2dOrdinary(output, m, n);
}
