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
    NumaNode*      p       = (NumaNode*)malloc(nthreads * sizeof(NumaNode));
    pthread_t*     threads = (pthread_t*)malloc(nthreads * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);
    int numanodes          = 8;
    int nthreads_each_node = nthreads / numanodes;
    for (int i = 0; i < nthreads; i++)
    {
        p[i].alloc           = i % numanodes;  // numa节点号
        p[i].numanodes       = numanodes;
        p[i].nthreads        = nthreads;
        p[i].nonzeros_in_row = A.nonzeros_in_row;
        p[i].M               = A.nrow / nthreads;  // 每个numa节点分配的行数
    }
    // 每个numa节点分配相等数量的线程，编号都从0开始
    for (int i = 0; i < nthreads_each_node; i++)
        for (int j = 0; j < numanodes; j++)
            p[i * numanodes + j].coreidx = i;
    p->sub_col_ind = (int**)malloc(sizeof(int*) * nthreads);
    p->sub_value   = (double**)malloc(sizeof(double*) * nthreads);
    p->X           = (double**)malloc(sizeof(double*) * nthreads);
    p->Y           = (double**)malloc(sizeof(double*) * nthreads);
    for (int i = 0; i < nthreads; i++)
    {
        p->sub_col_ind[i] = (int*)numa_alloc_onnode(sizeof(int) * A.nonzeros_in_row * A.nrow / nthreads, p[i].alloc);
        p->sub_value[i]   = (double*)numa_alloc_onnode(sizeof(double) * A.nonzeros_in_row * A.nrow / nthreads, p[i].alloc);
        p->X[i]           = (double*)numa_alloc_onnode(sizeof(double) * x.size, p[i].alloc);
        p->Y[i]           = (double*)numa_alloc_onnode(sizeof(double) * y.size, p[i].alloc);
    }
    int ntests = 1;
    for (int k = 0; k < ntests; k++)
    {
        for (int i = 0; i < nthreads; i++)
            pthread_create(&threads[i], &pthread_custom_attr, numaspmv, (void*)(p + i));
        for (int i = 0; i < nthreads; i++)
            pthread_join(threads[i], NULL);
    }
}

void* numaspmv(void* args)
{
    NumaNode* pn = (NumaNode*)args;
    int       me = pn->alloc;
    // numa_run_on_node(me);
    // int     M               = pn->M;
    // int     nthreads        = pn->nthreads;
    // int     numanodes       = pn->numanodes;
    // int     coreidx         = pn->coreidx;
    // int     threads_pernode = nthreads / numanodes;
    // int     task            = ceil((double)M / (double)threads_pernode);
    // int     start           = coreidx * task;
    // int     end             = (coreidx + 1) * task > M ? M : (coreidx + 1) * task;
    // int*    col_ind         = pn->sub_col_ind[me];
    // double* values          = pn->sub_value[me];
    // double* x               = pn->X[me];
    // double* y               = pn->Y[me];
    // for (int k = 0; k < pn->nonzeros_in_row; k++)
    // {
    //     for (int i = start; i < end; i++)
    //     {
    //         int col = col_ind[i + k * M];
    //         y[i] += values[i + k * M] * x[col];
    //     }
    // }
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
