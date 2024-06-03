/**
 * @file    : mat_vec.cpp
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Matrix and Vector Operations.
 */
#include <cassert>
#include <chrono>
#include <cmath>
#include <numa.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "mat_vec.h"
#include "mytime.h"
#include "numa_node.h"

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

void ell_matvec_numa(const ELL_Matrix& A, const Vector& x, const Vector& y, int nthreads)
{
    int numanodes          = 8;
    int nthreads_each_node = nthreads / numanodes;

    NumaNode*      p       = (NumaNode*)malloc(numanodes * sizeof(NumaNode));
    pthread_t*     threads = (pthread_t*)malloc(numanodes * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);
    for (int i = 0; i < numanodes; i++)
    {
        p[i].alloc           = i;  // numa节点号
        p[i].nthreads        = nthreads_each_node;
        p[i].nonzeros_in_row = A.nonzeros_in_row;
        p[i].M               = A.nrow / numanodes;  // 每个numa节点分配的行数
        p[i].sub_col_ind     = (int*)numa_alloc_onnode(sizeof(int) * A.nonzeros_in_row * p[i].M, p[i].alloc);
        p[i].sub_value       = (double*)numa_alloc_onnode(sizeof(double) * A.nonzeros_in_row * p[i].M, p[i].alloc);
        p[i].X               = (double*)numa_alloc_onnode(sizeof(double) * x.size, p[i].alloc);
        p[i].Y               = (double*)numa_alloc_onnode(sizeof(double) * p[i].M, p[i].alloc);
    }
    for (int i = 0; i < numanodes; i++)
    {
        memcpy(p[i].sub_col_ind, &A.col_ind[i * A.nonzeros_in_row * p[i].M], sizeof(int) * A.nonzeros_in_row * p[i].M);
        memcpy(p[i].sub_value, &A.values[i * A.nonzeros_in_row * p[i].M], sizeof(double) * A.nonzeros_in_row * p[i].M);
        memcpy(p[i].X, x.values, sizeof(double) * x.size);
        memcpy(p[i].Y, &y.values[i * p[i].M], sizeof(double) * p[i].M);
    }
    int    ntests  = 10;
    double t_begin = mytimer();
    for (int k = 0; k < ntests; k++)
    {
        for (int i = 0; i < numanodes; i++)
        {
            pthread_create(&threads[i], &pthread_custom_attr, numaspmv, (void*)&p[i]);
        }
        for (int i = 0; i < numanodes; i++)
        {
            int rc = pthread_join(threads[i], NULL);
        }
    }
    double t_end = mytimer();
    double t_avg = (t_end - t_begin) / ntests;
    printf("### ELL NUMA Compute Time = %.5f\n", t_avg);
}

void* numaspmv(void* args)
{
    NumaNode* pn = (NumaNode*)args;
    int       me = pn->alloc;
    numa_run_on_node(me);
    int     M        = pn->M;
    int     nthreads = pn->nthreads;
    int*    col_ind  = pn->sub_col_ind;
    double* values   = pn->sub_value;
    double* x        = pn->X;
    double* y        = pn->Y;

    for (int k = 0; k < pn->nonzeros_in_row; k++)
    {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
        for (int i = 0; i < pn->M; i++)
        {
            int col = col_ind[i + k * M];
            y[i] += values[i + k * M] * x[col];
        }
    }
    return NULL;
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
