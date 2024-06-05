#include "data_io.h"
#include "mat_vec.h"
#include "matrix.h"
#include "mmio.h"
#include "mytime.h"
#include "vec_vec.h"
#include "vector.h"

#include <math.h>
#include <numa.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define NUM_TEST 10

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: please input matrix file and nthreads\n");
        return -1;
    }
    char* file     = argv[1];
    int   nthreads = atoi(argv[2]);

#ifdef USE_OPENMP
    omp_set_num_threads(nthreads);
#endif

    COO_Matrix A;
    coo_read(file, A);
    int nrow, ncol;
    nrow = A.nrow;
    ncol = A.ncol;

    // CSR-SpMV
    CSR_Matrix B(A);
    Vector     x, y;
    x.Resize(ncol);
    y.Resize(nrow);
    x.FillRandom();
    y.Fill(0);
    double t_csr_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        csr_matvec(B, x, y);
    double t_csr_end = mytimer();
    double t_csr     = ((t_csr_end - t_csr_begin) * 1000.0 + (t_csr_end - t_csr_begin) / 1000.0) / NUM_TEST;
    printf("### CSR CPU GFLOPS = %.5f\n", 2 * A.nnz / t_csr / pow(10, 6));

    // CSC-SpMV
    CSC_Matrix C(A);
    y.Fill(0);
    double t_csc_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        csc_matvec(C, x, y);
    double t_csc_end = mytimer();
    double t_csc     = ((t_csc_end - t_csc_begin) * 1000.0 + (t_csc_end - t_csc_begin) / 1000.0) / NUM_TEST;
    printf("### CSC CPU GFLOPS = %.5f\n", 2 * A.nnz / t_csc / pow(10, 6));

    // COO-SpMV
    y.Fill(0);
    double t_coo_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        coo_matvec(A, x, y);
    double t_coo_end = mytimer();
    double t_coo     = ((t_coo_end - t_coo_begin) * 1000.0 + (t_coo_end - t_coo_begin) / 1000.0) / NUM_TEST;
    printf("### COO CPU GFLOPS = %.5f\n", 2 * A.nnz / t_coo / pow(10, 6));

    // ELL-SpMV
    ELL_Matrix D(A);
    y.Fill(0);
    double t_ell_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        ell_matvec(D, x, y);
    double t_ell_end = mytimer();
    double t_ell     = ((t_ell_end - t_ell_begin) * 1000.0 + (t_ell_end - t_ell_begin) / 1000.0) / NUM_TEST;
    printf("### ELL CPU GFLOPS = %.5f\n", 2 * A.nnz / t_ell / pow(10, 6));

    // DIA-SpMV
    DIA_Matrix E(A);
    y.Fill(0);
    double t_dia_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        dia_matvec(E, x, y);
    double t_dia_end = mytimer();
    double t_dia     = ((t_dia_end - t_dia_begin) * 1000.0 + (t_dia_end - t_dia_begin) / 1000.0) / NUM_TEST;
    printf("### DIA CPU GFLOPS = %.5f\n", 2 * A.nnz / t_dia / pow(10, 6));

    // ELL-SpMV-numa
    y.Fill(0);
    ell_matvec_numa(D, x, y, nthreads);

    return 0;
}