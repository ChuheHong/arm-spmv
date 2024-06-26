#include "vector.h"
#include <stdlib.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

Vector::Vector() : size(0), values(0) {}

Vector::Vector(int _n, double* _values) : size(_n), values(_values) {}

Vector::Vector(const Vector& x) : size(x.size), values(new double[x.size])
{
    double* xv = x.values;
    for (int i = 0; i < size; ++i)
        values[i] = xv[i];
}

Vector::~Vector()
{
    if (values)
        delete[] values;
}

Vector& Vector::operator=(double a)
{
    for (int i = 0; i < size; ++i)
        values[i] = a;
    return *this;
}

Vector& Vector::operator=(const Vector& x)
{
    Resize(x.size);
    double* xv = x.values;
    for (int i = 0; i < size; ++i)
        values[i] = xv[i];
    return *this;
}

void Vector::Free()
{
    if (values)
        delete[] values;
    size   = 0;
    values = 0;
}

void Vector::Resize(int n)
{
    if (values)
        delete[] values;
    size   = n;
    values = new double[n];
}

void Vector::Fill(double a) const
{
    for (int i = 0; i < size; ++i)
        values[i] = a;
}

void Vector::FillRandom() const
{
    for (int i = 0; i < size; ++i)
        values[i] = (double)rand() / RAND_MAX;
}

void Vector::Copy(const Vector& x) const
{
    double* xv = x.values;
    for (int i = 0; i < size; ++i)
        values[i] = xv[i];
}

void Vector::Scale(double a) const
{
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i)
        values[i] *= a;
}

void Vector::Shift(double a) const
{
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i)
        values[i] += a;
}

void Vector::AddScaled(double a, const Vector& x) const
{
    double* xv = x.values;
    if (a == 0)
        return;
    if (a == 1)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < size; ++i)
            values[i] += xv[i];
    }
    else if (a == -1)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < size; ++i)
            values[i] -= xv[i];
    }
    else
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < size; ++i)
            values[i] += a * xv[i];
    }
}

void Vector::Add2Scaled(double a, const Vector& x, double b, const Vector& y) const
{
    double* xv = x.values;
    double* yv = y.values;
    if (a == 0)
        AddScaled(b, y);
    else if (b == 0)
        AddScaled(a, x);
    else if (a == 1)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < size; ++i)
            values[i] += xv[i] + b * yv[i];
    }
    else if (b == 1)
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < size; ++i)
            values[i] += a * xv[i] + yv[i];
    }
    else
    {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < size; ++i)
            values[i] += a * xv[i] + b * yv[i];
    }
}