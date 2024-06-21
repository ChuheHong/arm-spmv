/**
 * @file    : data_io.cpp
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Matrix and Vector Read and Write.
 */

#include <stdio.h>

#include "data_io.h"
#include "mmio.h"

// ====================================================
// ###################### Vec R&W #####################
// ====================================================

void VectorRead(const char* filename, Vector& x)
{
    int   n;
    FILE* fp = fopen(filename, "r");
    fscanf(fp, "%d", &n);

    double* xv = new double[n];
    for (int i = 0; i < n; ++i)
        fscanf(fp, "\n%lg", &xv[i]);

    fclose(fp);

    x.Free();
    x.size   = n;
    x.values = xv;
}

void VectorWrite(const char* filename, const Vector& x)
{
    int     n  = x.size;
    double* xv = x.values;

    FILE* fp = fopen(filename, "w");

    fprintf(fp, "%d", n);

    for (int i = 0; i < n; ++i)
        fprintf(fp, "\n%20.16g", xv[i]);

    fclose(fp);
}

// ====================================================
// ###################### COO R&W #####################
// ====================================================
void COOMatrixRead(const char* filename, COOMatrix& A)
{
    int         ret_code;
    MM_typecode matcode;
    int         M, N, nz;
    FILE*       fp;

    printf("\tOpening matrix market file\n");
    if ((fp = fopen(filename, "r")) == NULL)
    {
        printf("***Failed to open MatrixMarket file %s ***\n", filename);
        exit(1);
    }

    printf("\tReading MatrixMarket banner\n");
    if (mm_read_banner(fp, &matcode) != 0)
    {
        printf("*** Could not process Matrix Market banner ***\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    printf("\tReading sparse matrix size...");
    if ((ret_code = mm_read_mtx_crd_size(fp, &N, &M, &nz)) != 0)
        exit(1);

    printf("\tAllocating memory for matrix\n");
    int*    II  = new int[nz];
    int*    JJ  = new int[nz];
    double* val = new double[nz];

    printf("\tReading matrix entries from file\n");
    for (int i = 0; i < nz; i++)
    {
        fscanf(fp, "%d %d %lg\n", &II[i], &JJ[i], &val[i]);
        II[i]--;
        JJ[i]--;
    }
    if (fp != stdin)
        fclose(fp);
    printf("### ROW=%d, COL=%d, NNZ=%d\n", N, M, nz);

    A.Free();
    A.nrow    = N;
    A.ncol    = M;
    A.nnz     = nz;
    A.row_ind = II;
    A.col_ind = JJ;
    A.values  = val;
    /*
        delete[] II;
        delete[] JJ;
        delete[] val;
    */
}

// ====================================================
// ###################### CSR R&W #####################
// ====================================================

void CSRMatrixRead(const char* filename, CSRMatrix& A)
{
    COOMatrix B;
    COOMatrixRead(filename, B);
    A = B;
}

// ====================================================
// ###################### CSC R&W #####################
// ====================================================

void CSCMatrixRead(const char* filename, CSCMatrix& A)
{
    COOMatrix B;
    COOMatrixRead(filename, B);
    A = B;
}

// ====================================================
// ###################### ELL R&W #####################
// ====================================================

void ELLMatrixRead(const char* filename, ELLMatrix& A)
{
    COOMatrix B;
    COOMatrixRead(filename, B);
    A = B;
}
