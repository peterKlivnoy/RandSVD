/*
 * SWIG Interface File for Hadamard Kernel
 * 
 * To generate Python wrapper:
 *   swig -python -o hadamard_wrap.c hadamard.i
 * 
 * This creates:
 *   - hadamard_wrap.c (C wrapper code)
 *   - hadamardKernel.py (Python interface)
 */

%module hadamardKernel

%{
#define SWIG_FILE_WITH_INIT
#include "hadamard_kernel.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

/* Map NumPy arrays to C pointers */
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *data, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double *input, int im, int in)};
%apply (double* ARGOUT_ARRAY2, int DIM1, int DIM2) {(double *output, int om, int on)};

/* Declare the function signatures */
void fwhtKernel2dOrdinary(double *data, int m, int n);
void fwhtKernel2dOrdinaryWrapper(double *input, int im, int in, 
                                  double *output, int om, int on);
