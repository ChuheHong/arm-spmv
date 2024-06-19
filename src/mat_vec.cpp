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
            int    row   = row_ind[j];
            double value = val[j] * xv[i];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            yv[row] += value;
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
    int numanodes     = numa_num_configured_nodes();
    int rows_per_node = y.size / nthreads;

    NumaNode4COO*  p       = (NumaNode4COO*)malloc(nthreads * sizeof(NumaNode4COO));
    pthread_t*     threads = (pthread_t*)malloc(nthreads * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    for (int i = 0; i < nthreads; i++)
    {
        int numa_node = i % numanodes;
        p[i].alloc    = numa_node;
        p[i].core_ind = i;
        int start_row = i * rows_per_node;
        int end_row   = (i + 1) * rows_per_node;
        if (i == nthreads - 1)
        {
            end_row = y.size;
        }

        int nnz_start_idx = -1;
        int nnz_end_idx   = A.nnz;
        for (int j = 0; j < A.nnz; j++)
        {
            if (A.row_ind[j] >= start_row && nnz_start_idx == -1)
            {
                nnz_start_idx = j;
            }
            if (A.row_ind[j] >= end_row)
            {
                nnz_end_idx = j;
                break;
            }
        }

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
        memset(p[i].Y, 0, sizeof(double) * (end_row - start_row));
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

    for (int i = 0; i < nthreads; i++)
    {
        numa_free(p[i].sub_row_ind, sizeof(int) * p[i].nnz);
        numa_free(p[i].sub_col_ind, sizeof(int) * p[i].nnz);
        numa_free(p[i].sub_values, sizeof(double) * p[i].nnz);
        numa_free(p[i].X, sizeof(double) * x.size);
        numa_free(p[i].Y, sizeof(double) * p[i].rows_per_node);
    }
    free(p);
    free(threads);
}

void csr_matvec_numa(const CSR_Matrix& A, const Vector& x, Vector& y, int nthreads)
{
    int numanodes       = numa_num_configured_nodes();
    int rows_per_thread = A.nrow / nthreads;

    NumaNode4CSR*  p       = (NumaNode4CSR*)malloc(nthreads * sizeof(NumaNode4CSR));
    pthread_t*     threads = (pthread_t*)malloc(nthreads * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    for (int i = 0; i < nthreads; i++)
    {
        int numa_node      = i % numanodes;
        p[i].alloc         = numa_node;
        p[i].core_ind      = i;
        p[i].start_row     = i * rows_per_thread;
        p[i].rows_per_node = (i == nthreads - 1) ? (A.nrow - p[i].start_row) : rows_per_thread;

        int start_row     = p[i].start_row;
        int end_row       = start_row + p[i].rows_per_node;
        int nnz_start_idx = A.row_ptr[start_row];
        int nnz_end_idx   = (end_row < A.nrow) ? A.row_ptr[end_row] : A.row_ptr[A.nrow];

        p[i].nnz         = nnz_end_idx - nnz_start_idx;
        p[i].sub_row_ptr = (int*)numa_alloc_onnode((p[i].rows_per_node + 1) * sizeof(int), p[i].alloc);
        p[i].sub_col_ind = (int*)numa_alloc_onnode(p[i].nnz * sizeof(int), p[i].alloc);
        p[i].sub_values  = (double*)numa_alloc_onnode(p[i].nnz * sizeof(double), p[i].alloc);
        p[i].X           = (double*)numa_alloc_onnode(x.size * sizeof(double), p[i].alloc);
        p[i].Y           = (double*)numa_alloc_onnode(p[i].rows_per_node * sizeof(double), p[i].alloc);

        for (int j = 0; j <= p[i].rows_per_node; j++)
        {
            p[i].sub_row_ptr[j] = A.row_ptr[start_row + j] - nnz_start_idx;
        }
        memcpy(p[i].sub_col_ind, &A.col_ind[nnz_start_idx], p[i].nnz * sizeof(int));
        memcpy(p[i].sub_values, &A.values[nnz_start_idx], p[i].nnz * sizeof(double));
        memcpy(p[i].X, x.values, x.size * sizeof(double));
        memset(p[i].Y, 0, p[i].rows_per_node * sizeof(double));
    }

    int    NTESTS  = 50;
    double t_begin = mytimer();
    for (int k = 0; k < NTESTS; k++)
    {
        for (int i = 0; i < nthreads; i++)
        {
            pthread_create(&threads[i], &pthread_custom_attr, numaspmv4csr, (void*)&p[i]);
        }
        for (int i = 0; i < nthreads; i++)
        {
            pthread_join(threads[i], NULL);
        }
    }
    double t_end = mytimer();
    double t_avg = ((t_end - t_begin) * 1000.0 + (t_end - t_begin) / 1000.0) / NTESTS;
    printf("### CSR NUMA GFLOPS = %.5f\n", 2 * A.row_ptr[A.nrow] / t_avg / pow(10, 6));

    for (int i = 0; i < nthreads; i++)
    {
        numa_free(p[i].sub_row_ptr, (p[i].rows_per_node + 1) * sizeof(int));
        numa_free(p[i].sub_col_ind, p[i].nnz * sizeof(int));
        numa_free(p[i].sub_values, p[i].nnz * sizeof(double));
        numa_free(p[i].X, x.size * sizeof(double));
        numa_free(p[i].Y, p[i].rows_per_node * sizeof(double));
    }
    free(p);
    free(threads);
}

void csc_matvec_numa(const CSC_Matrix& A, const Vector& x, Vector& y, int nthreads)
{
    int numanodes       = numa_num_configured_nodes();
    int cols_per_thread = A.ncol / nthreads;

    NumaNode4CSC*  p       = (NumaNode4CSC*)malloc(nthreads * sizeof(NumaNode4CSC));
    pthread_t*     threads = (pthread_t*)malloc(nthreads * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    for (int i = 0; i < nthreads; i++)
    {
        int numa_node      = i % numanodes;
        p[i].alloc         = numa_node;
        p[i].core_ind      = i;
        p[i].start_col     = i * cols_per_thread;
        p[i].cols_per_node = (i == nthreads - 1) ? (A.ncol - p[i].start_col) : cols_per_thread;

        int start_col     = p[i].start_col;
        int end_col       = start_col + p[i].cols_per_node;
        int nnz_start_idx = A.col_ptr[start_col];
        int nnz_end_idx   = A.col_ptr[end_col];

        p[i].nnz         = nnz_end_idx - nnz_start_idx;
        p[i].sub_col_ptr = (int*)numa_alloc_onnode((p[i].cols_per_node + 1) * sizeof(int), p[i].alloc);
        p[i].sub_row_ind = (int*)numa_alloc_onnode(p[i].nnz * sizeof(int), p[i].alloc);
        p[i].sub_values  = (double*)numa_alloc_onnode(p[i].nnz * sizeof(double), p[i].alloc);
        p[i].X           = (double*)numa_alloc_onnode(p[i].cols_per_node * sizeof(double), p[i].alloc);
        p[i].Y           = (double*)numa_alloc_onnode(y.size * sizeof(double), p[i].alloc);

        for (int j = 0; j <= p[i].cols_per_node; j++)
        {
            p[i].sub_col_ptr[j] = A.col_ptr[start_col + j] - nnz_start_idx;
        }
        memcpy(p[i].sub_row_ind, &A.row_ind[nnz_start_idx], p[i].nnz * sizeof(int));
        memcpy(p[i].sub_values, &A.values[nnz_start_idx], p[i].nnz * sizeof(double));
        memcpy(p[i].X, &x.values[start_col], p[i].cols_per_node * sizeof(double));
        memset(p[i].Y, 0, y.size * sizeof(double));
    }

    int    NTESTS  = 50;
    double t_begin = mytimer();
    for (int k = 0; k < NTESTS; k++)
    {
        for (int i = 0; i < nthreads; i++)
        {
            pthread_create(&threads[i], &pthread_custom_attr, numaspmv4csc, (void*)&p[i]);
        }
        for (int i = 0; i < nthreads; i++)
        {
            pthread_join(threads[i], NULL);
        }
    }
    double t_end = mytimer();

    // Reduce Y from all threads
    memset(y.values, 0, y.size * sizeof(double));
    for (int i = 0; i < nthreads; i++)
    {
        for (int j = 0; j < y.size; j++)
        {
            y.values[j] += p[i].Y[j];
        }
    }

    double t_avg = ((t_end - t_begin) * 1000.0 + (t_end - t_begin) / 1000.0) / NTESTS;
    printf("### CSC NUMA GFLOPS = %.5f\n", 2 * A.col_ptr[A.ncol] / t_avg / pow(10, 6));

    for (int i = 0; i < nthreads; i++)
    {
        numa_free(p[i].sub_col_ptr, (p[i].cols_per_node + 1) * sizeof(int));
        numa_free(p[i].sub_row_ind, p[i].nnz * sizeof(int));
        numa_free(p[i].sub_values, p[i].nnz * sizeof(double));
        numa_free(p[i].X, p[i].cols_per_node * sizeof(double));
        numa_free(p[i].Y, y.size * sizeof(double));
    }
    free(p);
    free(threads);
}

void ell_matvec_numa(const ELL_Matrix& A, const Vector& x, Vector& y, int nthreads)
{
    int numanodes       = numa_num_configured_nodes();
    int rows_per_thread = A.nrow / nthreads;

    NumaNode4ELL*  p       = (NumaNode4ELL*)malloc(nthreads * sizeof(NumaNode4ELL));
    pthread_t*     threads = (pthread_t*)malloc(nthreads * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    for (int i = 0; i < nthreads; i++)
    {
        int numa_node        = i % numanodes;
        p[i].alloc           = numa_node;
        p[i].core_ind        = i;
        p[i].rows_per_node   = (i == nthreads - 1) ? (A.nrow - i * rows_per_thread) : rows_per_thread;
        p[i].nonzeros_in_row = A.nonzeros_in_row;

        int start_row = i * rows_per_thread;
        int end_row   = start_row + p[i].rows_per_node;

        p[i].sub_col_ind = (int*)numa_alloc_onnode(p[i].rows_per_node * A.nonzeros_in_row * sizeof(int), p[i].alloc);
        p[i].sub_values  = (double*)numa_alloc_onnode(p[i].rows_per_node * A.nonzeros_in_row * sizeof(double), p[i].alloc);
        p[i].X           = (double*)numa_alloc_onnode(x.size * sizeof(double), p[i].alloc);
        p[i].Y           = (double*)numa_alloc_onnode(p[i].rows_per_node * sizeof(double), p[i].alloc);

        memcpy(p[i].sub_col_ind, &A.col_ind[start_row * A.nonzeros_in_row], p[i].rows_per_node * A.nonzeros_in_row * sizeof(int));
        memcpy(p[i].sub_values, &A.values[start_row * A.nonzeros_in_row], p[i].rows_per_node * A.nonzeros_in_row * sizeof(double));
        memcpy(p[i].X, x.values, x.size * sizeof(double));
        memset(p[i].Y, 0, p[i].rows_per_node * sizeof(double));
    }

    int    NTESTS  = 50;
    double t_begin = mytimer();
    for (int k = 0; k < NTESTS; k++)
    {
        for (int i = 0; i < nthreads; i++)
        {
            pthread_create(&threads[i], &pthread_custom_attr, numaspmv4ell, (void*)&p[i]);
        }
        for (int i = 0; i < nthreads; i++)
        {
            pthread_join(threads[i], NULL);
        }
    }
    double t_end = mytimer();
    double t_avg = ((t_end - t_begin) * 1000.0 + (t_end - t_begin) / 1000.0) / NTESTS;
    printf("### ELL NUMA GFLOPS = %.5f\n", 2 * A.nrow * A.nonzeros_in_row / t_avg / pow(10, 6));

    for (int i = 0; i < nthreads; i++)
    {
        numa_free(p[i].sub_col_ind, p[i].rows_per_node * A.nonzeros_in_row * sizeof(int));
        numa_free(p[i].sub_values, p[i].rows_per_node * A.nonzeros_in_row * sizeof(double));
        numa_free(p[i].X, x.size * sizeof(double));
        numa_free(p[i].Y, p[i].rows_per_node * sizeof(double));
    }
    free(p);
    free(threads);
}

void dia_matvec_numa(const DIA_Matrix& A, const Vector& x, Vector& y, int nthreads)
{
    int numanodes       = numa_num_configured_nodes();
    int rows_per_thread = A.nrow / nthreads;

    NumaNode4DIA*  p       = (NumaNode4DIA*)malloc(nthreads * sizeof(NumaNode4DIA));
    pthread_t*     threads = (pthread_t*)malloc(nthreads * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    for (int i = 0; i < nthreads; i++)
    {
        int numa_node      = i % numanodes;
        p[i].alloc         = numa_node;
        p[i].core_ind      = i;
        p[i].start_row     = i * rows_per_thread;
        p[i].rows_per_node = (i == nthreads - 1) ? (A.nrow - p[i].start_row) : rows_per_thread;
        p[i].ndiags        = A.ndiags;
        p[i].offsets       = A.offsets;
        p[i].values        = (double*)numa_alloc_onnode(p[i].rows_per_node * A.ndiags * sizeof(double), numa_node);
        p[i].X             = (double*)numa_alloc_onnode(x.size * sizeof(double), numa_node);
        p[i].Y             = (double*)numa_alloc_onnode(p[i].rows_per_node * sizeof(double), numa_node);
        memcpy(p[i].values, A.values + p[i].start_row * A.ndiags, p[i].rows_per_node * A.ndiags * sizeof(double));
        memcpy(p[i].X, x.values, x.size * sizeof(double));
        memset(p[i].Y, 0, p[i].rows_per_node * sizeof(double));
    }

    int    NTESTS  = 50;
    double t_begin = mytimer();
    for (int k = 0; k < NTESTS; k++)
    {
        for (int i = 0; i < nthreads; i++)
        {
            pthread_create(&threads[i], &pthread_custom_attr, numaspmv4dia, (void*)&p[i]);
        }
        for (int i = 0; i < nthreads; i++)
        {
            pthread_join(threads[i], NULL);
        }
    }
    double t_end = mytimer();
    double t_avg = ((t_end - t_begin) * 1000.0 + (t_end - t_begin) / 1000.0) / NTESTS;
    printf("### DIA NUMA GFLOPS = %.5f\n", 2 * A.nnz / t_avg / pow(10, 6));

    for (int i = 0; i < nthreads; i++)
    {
        for (int j = 0; j < p[i].rows_per_node; j++)
        {
            y.values[p[i].start_row + j] = p[i].Y[j];
        }
        numa_free(p[i].X, x.size * sizeof(double));
        numa_free(p[i].Y, p[i].rows_per_node * sizeof(double));
        numa_free(p[i].values, p[i].rows_per_node * A.ndiags * sizeof(double));
    }
    free(p);
    free(threads);
}

void* numaspmv4coo(void* args)
{
    NumaNode4COO* pn = (NumaNode4COO*)args;
    int           me = pn->alloc;
    numa_run_on_node(me);
    int     nnz     = pn->nnz;
    int*    row_ind = pn->sub_row_ind;
    int*    col_ind = pn->sub_col_ind;
    double* values  = pn->sub_values;
    double* x       = pn->X;
    double* y       = pn->Y;

    for (int i = 0; i < nnz; i++)
    {
        int row = row_ind[i] - pn->start_row;
        int col = col_ind[i];
        y[row] += values[i] * x[col];
    }
    return NULL;
}

void* numaspmv4csr(void* args)
{
    NumaNode4CSR* pn = (NumaNode4CSR*)args;
    int           me = pn->alloc;
    numa_run_on_node(me);
    int*    row_ptr   = pn->sub_row_ptr;
    int*    col_ind   = pn->sub_col_ind;
    double* values    = pn->sub_values;
    double* x         = pn->X;
    double* y         = pn->Y;
    int     start_row = pn->start_row;
    int     end_row   = start_row + pn->rows_per_node;

    for (int i = 0; i < pn->rows_per_node; i++)
    {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            sum += values[j] * x[col_ind[j]];
        }
        y[i] += sum;
    }
    return NULL;
}

void* numaspmv4csc(void* args)
{
    NumaNode4CSC* pn = (NumaNode4CSC*)args;
    int           me = pn->alloc;
    numa_run_on_node(me);
    int*    col_ptr       = pn->sub_col_ptr;
    int*    row_ind       = pn->sub_row_ind;
    double* values        = pn->sub_values;
    double* x             = pn->X;
    double* y             = pn->Y;
    int     cols_per_node = pn->cols_per_node;
    int     start_col     = pn->start_col;

    for (int col = 0; col < cols_per_node; col++)
    {
        for (int idx = col_ptr[col]; idx < col_ptr[col + 1]; idx++)
        {
            int row = row_ind[idx];
            y[row] += values[idx] * x[col];
        }
    }
    return NULL;
}

void* numaspmv4ell(void* args)
{
    NumaNode4ELL* pn = (NumaNode4ELL*)args;
    int           me = pn->alloc;
    numa_run_on_node(me);
    int*    col_ind         = pn->sub_col_ind;
    double* values          = pn->sub_values;
    double* x               = pn->X;
    double* y               = pn->Y;
    int     rows_per_node   = pn->rows_per_node;
    int     nonzeros_in_row = pn->nonzeros_in_row;

    for (int i = 0; i < rows_per_node; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < nonzeros_in_row; j++)
        {
            sum += values[i * nonzeros_in_row + j] * x[col_ind[i * nonzeros_in_row + j]];
        }
        y[i] = sum;
    }
    return NULL;
}

void* numaspmv4dia(void* args)
{
    NumaNode4DIA* pn = (NumaNode4DIA*)args;
    int           me = pn->alloc;
    numa_run_on_node(me);

    int     nrow    = pn->rows_per_node;
    int     ndiags  = pn->ndiags;
    int*    offsets = pn->offsets;
    double* values  = pn->values;
    double* xv      = pn->X;
    double* yv      = pn->Y;

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

    return NULL;
}
