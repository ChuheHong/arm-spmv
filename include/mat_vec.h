/**
 * @file    : mat_vec.h
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Matrix and Vector Operations.
 */
#ifndef MAT_VEC_H
#define MAT_VEC_H

#include "matrix.h"
#include "vector.h"

void coo_matvec(const COO_Matrix& A, const Vector& x, Vector& y);
void csr_matvec(const CSR_Matrix& A, const Vector& x, Vector& y);
void csc_matvec(const CSC_Matrix& A, const Vector& x, Vector& y);
void ell_matvec(const ELL_Matrix& A, const Vector& x, Vector& y);
void dia_matvec(const DIA_Matrix& A, const Vector& x, Vector& y);
void ell_matvec_numa(const ELL_Matrix& A, const Vector& x, Vector& y, int nthreads);

void csr_symgs(const CSR_Matrix& A, const Vector& r, const Vector& x);
void ell_symgs(const ELL_Matrix& A, const Vector& r, const Vector& x);

void* numaspmv(void* args);

#endif