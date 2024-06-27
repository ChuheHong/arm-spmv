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

#define NUM_TEST 50

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

    COOMatrix A;
    Vector    x, y;
    COOMatrixRead(file, A);
    int nrow, ncol;
    nrow = A.nrow;
    ncol = A.ncol;
    x.Resize(ncol);
    y.Resize(nrow);
    x.FillRandom();

    Vector y_ref;
    y_ref.Resize(nrow);
    y_ref.Fill(0);
    // Check correctness
    for (int k = 0; k < NUM_TEST; k++)
    {
        for (int i = 0; i < A.nnz; i++)
        {
            y_ref.values[A.row_ind[i]] += A.values[i] * x.values[A.col_ind[i]];
        }
    }

    // COO-SpMV
    y.Fill(0);
    double t_coo_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        COOMatirxMatVector(A, x, y);
    double t_coo_end = mytimer();
    double t_coo     = ((t_coo_end - t_coo_begin) * 1000.0 + (t_coo_end - t_coo_begin) / 1000.0) / NUM_TEST;
    printf("### COO CPU GFLOPS = %.5f\n", 2 * A.nnz / t_coo / pow(10, 6));

    // CSR-SpMV
    CSRMatrix B(A);
    y.Fill(0);
    double t_csr_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        CSRMatrixMatVector(B, x, y);
    double t_csr_end = mytimer();
    double t_csr     = ((t_csr_end - t_csr_begin) * 1000.0 + (t_csr_end - t_csr_begin) / 1000.0) / NUM_TEST;
    printf("### CSR CPU GFLOPS = %.5f\n", 2 * A.nnz / t_csr / pow(10, 6));

    // CSC-SpMV
    CSCMatrix C(A);
    y.Fill(0);
    double t_csc_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        CSCMatrixMatVector(C, x, y);
    double t_csc_end = mytimer();
    double t_csc     = ((t_csc_end - t_csc_begin) * 1000.0 + (t_csc_end - t_csc_begin) / 1000.0) / NUM_TEST;
    printf("### CSC CPU GFLOPS = %.5f\n", 2 * A.nnz / t_csc / pow(10, 6));

    // ELL-SpMV
    ELLMatrix D(A);
    y.Fill(0);
    double t_ell_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        ELLMatrixMatVector(D, x, y);
    double t_ell_end = mytimer();
    double t_ell     = ((t_ell_end - t_ell_begin) * 1000.0 + (t_ell_end - t_ell_begin) / 1000.0) / NUM_TEST;
    printf("### ELL CPU GFLOPS = %.5f\n", 2 * A.nnz / t_ell / pow(10, 6));

    // DIA-SpMV
    DIAMatrix E(B);
    y.Fill(0);
    double t_dia_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        DIAMatrixMatVector(E, x, y);
    double t_dia_end = mytimer();
    double t_dia     = ((t_dia_end - t_dia_begin) * 1000.0 + (t_dia_end - t_dia_begin) / 1000.0) / NUM_TEST;
    printf("### DIA CPU GFLOPS = %.5f\n", 2 * A.nnz / t_dia / pow(10, 6));

    printf("#############################\n");

    // COO-SpMV-numa
    y.Fill(0);
    COOMatrixMatVectorNuma(A, x, y, nthreads);

    // CSR-SpMV-numa
    y.Fill(0);
    CSRMatrixMatVectorNuma(B, x, y, nthreads);

    // CSC-SpMV-numa
    y.Fill(0);
    CSCMatrixMatVectorNuma(C, x, y, nthreads);

    // ELL-SpMV-numa
    y.Fill(0);
    ELLMatrixMatVectorNuma(D, x, y, nthreads);

    // DIA-SpMV-numa
    y.Fill(0);
    DIAMatrixMatVectorNuma(E, x, y, nthreads);

    return 0;
}