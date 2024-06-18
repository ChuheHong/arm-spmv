/**
 * @file    : mat_vec.cpp
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Matrix and Vector Operations.
 */
#include <algorithm>
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

void coo_matvec(const COO_Matrix& A, const Vector& x, Vector& y)
{
    int     nrow    = A.nrow;
    int     nnz     = A.nnz;
    int*    row_ind = A.row_ind;
    int*    col_ind = A.col_ind;
    double* val     = A.values;

    double* xv = x.values;
    double* yv = y.values;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nnz; i++)
    {
        int row = row_ind[i];
        int col = col_ind[i];
#ifdef USE_OPENMP
#pragma omp atmoic
#endif
        yv[row] += val[i] * xv[col];
    }
    return;
}

void csr_matvec(const CSR_Matrix& A, const Vector& x, Vector& y)
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

void csc_matvec(const CSC_Matrix& A, const Vector& x, Vector& y)
{
    int     ncol    = A.ncol;
    int*    row_ind = A.row_ind;
    int*    col_ptr = A.col_ptr;
    double* val     = A.values;

    double* xv = x.values;
    double* yv = y.values;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < ncol; i++)
    {
        for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
        {
            yv[row_ind[j]] += val[j] * xv[i];
        }
    }
    return;
}

void ell_matvec(const ELL_Matrix& A, const Vector& x, Vector& y)
{
    int     nrow            = A.nrow;
    int     nonzeros_in_row = A.nonzeros_in_row;
    int*    col_ind         = A.col_ind;
    double* val             = A.values;

    double* xv = x.values;
    double* yv = y.values;

    for (int k = 0; k < nonzeros_in_row; k++)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < nrow; i++)
        {
            int curCol;
            curCol = col_ind[i + k * nrow];
            yv[i] += val[i + k * nrow] * xv[curCol];
        }
    }

    return;
}

void dia_matvec(const DIA_Matrix& A, const Vector& x, Vector& y)
{
    int     nrow    = A.nrow;
    int     ndiags  = A.ndiags;
    int*    offsets = A.offsets;
    double* values  = A.values;
    double* xv      = x.values;
    double* yv      = y.values;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nrow; ++i)
    {
        for (int d = 0; d < ndiags; ++d)
        {
            int j = i + offsets[d];
            if (j >= 0 && j < nrow)
            {
                yv[i] += values[i * ndiags + d] * xv[j];
            }
        }
    }
}

void coo_matvec_numa(const COO_Matrix& A, const Vector& x, Vector& y, int nthreads)
{
    int numanodes          = 8;
    int nthreads_each_node = nthreads / numanodes;
    int rows_per_node      = y.size / numanodes;

    NumaNode4COO*  p       = (NumaNode4COO*)malloc(numanodes * sizeof(NumaNode4COO));
    pthread_t*     threads = (pthread_t*)malloc(numanodes * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    for (int i = 0; i < numanodes; i++)
    {
        p[i].alloc    = i;
        p[i].nthreads = nthreads_each_node;
        int start_row = i * rows_per_node;
        int end_row   = (i + 1) * rows_per_node;
        if (i == numanodes - 1)
        {
            end_row = y.size;
        }

        int* row_begin     = A.row_ind;
        int* row_end       = A.row_ind + A.nnz;
        int* nnz_start_ptr = std::lower_bound(row_begin, row_end, start_row);
        int* nnz_end_ptr   = std::lower_bound(row_begin, row_end, end_row);
        int  nnz_start_idx = nnz_start_ptr - row_begin;
        int  nnz_end_idx   = nnz_end_ptr - row_begin;

        p[i].nnz           = nnz_end_idx - nnz_start_idx;
        p[i].start_row     = start_row;
        p[i].rows_per_node = end_row - start_row;
        p[i].sub_row_ind   = (int*)numa_alloc_onnode(sizeof(int) * p[i].nnz, p[i].alloc);
        p[i].sub_col_ind   = (int*)numa_alloc_onnode(sizeof(int) * p[i].nnz, p[i].alloc);
        p[i].sub_values    = (double*)numa_alloc_onnode(sizeof(double) * p[i].nnz, p[i].alloc);
        p[i].X             = (double*)numa_alloc_onnode(sizeof(double) * x.size, p[i].alloc);
        p[i].Y             = (double*)numa_alloc_onnode(sizeof(double) * rows_per_node, p[i].alloc);

        memcpy(p[i].sub_row_ind, &A.row_ind[nnz_start_idx], sizeof(int) * p[i].nnz);
        memcpy(p[i].sub_col_ind, &A.col_ind[nnz_start_idx], sizeof(int) * p[i].nnz);
        memcpy(p[i].sub_values, &A.values[nnz_start_idx], sizeof(double) * p[i].nnz);
        memcpy(p[i].X, x.values, sizeof(double) * x.size);
        memcpy(p[i].Y, &y.values[start_row], sizeof(double) * (end_row - start_row));
    }

    int    NTESTS  = 50;
    double t_begin = mytimer();
    for (int k = 0; k < NTESTS; k++)
    {
        for (int i = 0; i < numanodes; i++)
        {
            pthread_create(&threads[i], &pthread_custom_attr, numaspmv4coo, (void*)&p[i]);
        }
        for (int i = 0; i < numanodes; i++)
        {
            pthread_join(threads[i], NULL);
        }
    }
    double t_end = mytimer();
    double t_avg = ((t_end - t_begin) * 1000.0 + (t_end - t_begin) / 1000.0) / NTESTS;
    printf("### COO NUMA GFLOPS = %.5f\n", 2 * A.nnz / t_avg / pow(10, 6));
}

void csr_matvec_numa(const CSR_Matrix& A, const Vector& x, Vector& y, int nthreads) {}

void csc_matvec_numa(const CSC_Matrix& A, const Vector& x, Vector& y, int nthreads) {}

void ell_matvec_numa(const ELL_Matrix& A, const Vector& x, Vector& y, int nthreads)
{
    int numanodes          = 8;
    int nthreads_each_node = nthreads / numanodes;

    NumaNode4ELL*  p       = (NumaNode4ELL*)malloc(numanodes * sizeof(NumaNode4ELL));
    pthread_t*     threads = (pthread_t*)malloc(numanodes * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);
    for (int i = 0; i < numanodes; i++)
    {
        p[i].alloc           = i;  // numa节点号
        p[i].nthreads        = nthreads_each_node;
        p[i].nonzeros_in_row = A.nonzeros_in_row;
        p[i].rows_per_node   = A.nrow / numanodes;  // 每个numa节点分配的行数
        p[i].sub_col_ind     = (int*)numa_alloc_onnode(sizeof(int) * A.nonzeros_in_row * p[i].rows_per_node, p[i].alloc);
        p[i].sub_values      = (double*)numa_alloc_onnode(sizeof(double) * A.nonzeros_in_row * p[i].rows_per_node, p[i].alloc);
        p[i].X               = (double*)numa_alloc_onnode(sizeof(double) * x.size, p[i].alloc);
        p[i].Y               = (double*)numa_alloc_onnode(sizeof(double) * p[i].rows_per_node, p[i].alloc);
    }
    for (int i = 0; i < numanodes; i++)
    {
        memcpy(p[i].sub_col_ind, &A.col_ind[i * A.nonzeros_in_row * p[i].rows_per_node], sizeof(int) * A.nonzeros_in_row * p[i].rows_per_node);
        memcpy(p[i].sub_values, &A.values[i * A.nonzeros_in_row * p[i].rows_per_node], sizeof(double) * A.nonzeros_in_row * p[i].rows_per_node);
        memcpy(p[i].X, x.values, sizeof(double) * x.size);
        memcpy(p[i].Y, &y.values[i * p[i].rows_per_node], sizeof(double) * p[i].rows_per_node);
    }
    int    NTESTS  = 50;
    double t_begin = mytimer();
    for (int k = 0; k < NTESTS; k++)
    {
        for (int i = 0; i < numanodes; i++)
        {
            pthread_create(&threads[i], &pthread_custom_attr, numaspmv4ell, (void*)&p[i]);
        }
        for (int i = 0; i < numanodes; i++)
        {
            pthread_join(threads[i], NULL);
        }
    }
    double t_end = mytimer();
    double t_avg = ((t_end - t_begin) * 1000.0 + (t_end - t_begin) / 1000.0) / NTESTS;
    printf("### ELL NUMA GFLOPS = %.5f\n", 2 * A.nnz / t_avg / pow(10, 6));
}

void dia_matvec_numa(const DIA_Matrix& A, const Vector& x, Vector& y, int nthreads) {}

void* numaspmv4coo(void* args)
{
    NumaNode4COO* pn = (NumaNode4COO*)args;
    int           me = pn->alloc;
    numa_run_on_node(me);
    int     nnz           = pn->nnz;
    int     nthreads      = pn->nthreads;
    int*    row_ind       = pn->sub_row_ind;
    int*    col_ind       = pn->sub_col_ind;
    int     rows_per_node = pn->rows_per_node;
    int     start_row     = pn->start_row;
    double* values        = pn->sub_values;
    double* x             = pn->X;
    double* y             = pn->Y;

#ifdef USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for (int i = 0; i < nnz; i++)
    {
        int row = row_ind[i] - start_row;  // Adjust row index to local y
        int col = col_ind[i];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
        y[row] += values[i] * x[col];
    }
    return NULL;
}

void* numaspmv4csr(void* args) {}

void* numaspmv4csc(void* args) {}

void* numaspmv4ell(void* args)
{
    NumaNode4ELL* pn = (NumaNode4ELL*)args;
    int           me = pn->alloc;
    numa_run_on_node(me);
    int     nrow     = pn->rows_per_node;
    int     nthreads = pn->nthreads;
    int*    col_ind  = pn->sub_col_ind;
    double* values   = pn->sub_values;
    double* x        = pn->X;
    double* y        = pn->Y;

    for (int k = 0; k < pn->nonzeros_in_row; k++)
    {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
        for (int i = 0; i < pn->rows_per_node; i++)
        {
            int col = col_ind[i + k * nrow];
            y[i] += values[i + k * nrow] * x[col];
        }
    }
    return NULL;
}

void* numaspmv4dia(void* args);