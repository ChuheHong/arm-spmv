/**
 * @file    : vec_operate.h
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Vector operations.
 */
#ifndef VEC_VEC_H
#define VEC_VEC_H

#include "vector.h"

double vec_dot(const Vector& x, const Vector& y);
void   vec_axpby(double alpha, const Vector& x, double beta, const Vector& y, const Vector& w);

#endif