/**
 * @file    : vec_operate.cpp
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Vector operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vec_vec.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

// ====================================================
// #################### Vec&Vec CPU ###################
// ====================================================

double vec_dot(const Vector& x, const Vector& y)
{
    int     n  = x.size;
    double* xv = x.values;
    double* yv = y.values;

    double result = 0.0;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+ : result)
#endif
    for (int i = 0; i < n; ++i)
        result += xv[i] * yv[i];

    return result;
}

void vec_axpby(double alpha, const Vector& x, double beta, const Vector& y, const Vector& w)
{
    int     n  = w.size;
    double* xv = x.values;
    double* yv = y.values;
    double* wv = w.values;

    if (alpha == 0)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; ++i)
            wv[i] = beta * yv[i];
    }
    else if (beta == 0)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; ++i)
            wv[i] = alpha * xv[i];
    }
    else if (alpha == 1)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; ++i)
            wv[i] = beta * yv[i] + xv[i];
    }
    else if (alpha == -1)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; ++i)
            wv[i] = beta * yv[i] - xv[i];
    }
    else if (beta == 1)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; ++i)
            wv[i] = alpha * xv[i] + yv[i];
    }
    else if (beta == -1)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; ++i)
            wv[i] = alpha * xv[i] - yv[i];
    }
    else
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; ++i)
            wv[i] = alpha * xv[i] + beta * yv[i];
    }
}