/**
 * @file    : mat_vec.cpp
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Matrix and Vector Operations.
 */
#include <cassert>
#include <stdio.h>
#include <string.h>

#include "mat_vec.h"
#include "mytime.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

void coo_matvec(const COO_Matrix& A, const Vector& x, const Vector& y)
{
    int     nrow    = A.nrow;
    int     nnz     = A.nnz;
    int*    row_ind = A.row_ind;
    int*    col_ind = A.col_ind;
    double* val     = A.values;

    double* xv = x.values;
    double* yv = y.values;

    for (int i = 0; i < nrow; i++)
        yv[i] = 0.0;

    for (int k = 0; k < nnz; k++)
    {
        yv[row_ind[k]] += val[k] * xv[col_ind[k]];
    }
    return;
}

void csr_matvec(const CSR_Matrix& A, const Vector& x, const Vector& y)
{
    int     nrow    = A.nrow;
    int*    row_ptr = A.row_ptr;
    int*    col_ind = A.col_ind;
    double* val     = A.values;

    double* xv = x.values;
    double* yv = y.values;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nrow; i++)
    {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            sum += val[j] * xv[col_ind[j]];
        }
        yv[i] = sum;
    }
    return;
}

void csc_matvec(const CSC_Matrix& A, const Vector& x, const Vector& y)
{
    int     ncol    = A.ncol;
    int*    row_ind = A.row_ind;
    int*    col_ptr = A.col_ptr;
    double* val     = A.values;

    double* xv = x.values;
    double* yv = y.values;

    for (int i = 0; i < ncol; i++)
    {
        for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
        {
            yv[row_ind[j]] += val[j] * xv[i];
        }
    }
    return;
}

void ell_matvec(const ELL_Matrix& A, const Vector& x, const Vector& y)
{
    int     nrow            = A.nrow;
    int     nonzeros_in_row = A.nonzeros_in_row;
    int*    col_ind         = A.col_ind;
    double* val             = A.values;

    double* xv = x.values;
    double* yv = y.values;

    for (int k = 0; k < nonzeros_in_row; k++)
    {
        for (int i = 0; i < nrow; i++)
        {
            int curCol;
            curCol = col_ind[i + k * nrow];
            yv[i] += val[i + k * nrow] * xv[curCol];
        }
    }

    return;
}

void csr_symgs(const CSR_Matrix& A, const Vector& r, const Vector& x)
{
    assert(x.size == A.ncol);
    int     nrow    = A.nrow;
    int*    row_ptr = A.row_ptr;
    int*    col_ind = A.col_ind;
    double* val     = A.values;
    double* diag    = A.diagonal;

    double* rv = r.values;
    double* xv = x.values;

    // the forward sweep
    for (int i = 0; i < nrow; i++)
    {
        double sum = rv[i];
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            sum -= val[j] * xv[col_ind[j]];
        }
        sum += xv[i] * diag[i];
        xv[i] = sum / diag[i];
    }

    // the backward sweep
    for (int i = nrow - 1; i >= 0; i--)
    {
        double sum = rv[i];
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            sum -= val[j] * xv[col_ind[j]];
        }
        sum += xv[i] * diag[i];
        xv[i] = sum / diag[i];
    }
    return;
}

void ell_symgs(const ELL_Matrix& A, const Vector& r, const Vector& x)
{
    assert(x.size == A.ncol);
    int max_col = 1024 * 128 * 6;  // 786432
    if (A.ncol > max_col)
    {
        exit(-1);
    }

    int     nrow            = A.nrow;
    int     nonzeros_in_row = A.nonzeros_in_row;
    int*    col_ind         = A.col_ind;
    double* val             = A.values;
    double* diag            = A.diagonal;

    double* rv = r.values;
    double* xv = x.values;

    // the forward sweep
    for (int i = 0; i < nrow; i++)
    {
        double currentDiagonal = diag[i];
        double sum             = rv[i];  // RHS value
        int    curCol;

        for (int j = 0; j < nonzeros_in_row; j++)
        {
            curCol = col_ind[i + j * nrow];
            sum -= val[i + j * nrow] * xv[curCol];
        }
        sum += xv[i] * currentDiagonal;  // Remove diagonal contribution from previous loop
        xv[i] = sum / currentDiagonal;
    }
    /*
        // the backward sweep
        for(int i = nrow-1; i >= 0; i--){
            double currentDiagonal = diag[i];
            double sum = rv[i];  // RHS value
            int curCol;

            for(int j = 0; j < nonzeros_in_row; j++){
                curCol = col_ind[i + j * nrow];
                sum -= val[i + j * nrow] * xv[curCol];
            }
            sum  += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop
            xv[i] = sum / currentDiagonal;
        }
    */
    return;
}
