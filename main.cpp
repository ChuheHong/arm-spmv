#include "data_io.h"
#include "mat_vec.h"
#include "matrix.h"
#include "mmio.h"
#include "mytime.h"
#include "vec_vec.h"
#include "vector.h"

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
    double t_csr     = (t_csr_end - t_csr_begin) / NUM_TEST;
    printf("### CSR CPU Compute Time = %.5f\n", t_csr);

    // ELL-SpMV
    ELL_Matrix C(A);
    y.Fill(0);
    double t_ell_begin = mytimer();
    for (int i = 0; i < NUM_TEST; i++)
        ell_matvec(C, x, y);
    double t_ell_end = mytimer();
    double t_ell     = (t_ell_end - t_ell_begin) / NUM_TEST;
    printf("### ELL CPU Compute Time = %.5f\n", t_ell);

    

    return 0;
}