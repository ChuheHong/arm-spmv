#ifndef MAT_VEC_H
#define MAT_VEC_H

#include "matrix.h"
#include "vector.h"

void COOMatirxMatVector(const COOMatrix& A, const Vector& x, Vector& y);
void CSRMatrixMatVector(const CSRMatrix& A, const Vector& x, Vector& y);
void CSCMatrixMatVector(const CSCMatrix& A, const Vector& x, Vector& y);
void ELLMatrixMatVector(const ELLMatrix& A, const Vector& x, Vector& y);
void DIAMatrixMatVector(const DIAMatrix& A, const Vector& x, Vector& y);

void COOMatrixMatVectorNuma(const COOMatrix& A, const Vector& x, Vector& y, int nthreads);
void CSRMatrixMatVectorNuma(const CSRMatrix& A, const Vector& x, Vector& y, int nthreads);
void CSCMatrixMatVectorNuma(const CSCMatrix& A, const Vector& x, Vector& y, int nthreads);
void ELLMatrixMatVectorNuma(const ELLMatrix& A, const Vector& x, Vector& y, int nthreads);
void DIAMatrixMatVectorNuma(const DIAMatrix& A, const Vector& x, Vector& y, int nthreads);

void* COOMatrixMatVectorNumaThread(void* args);
void* CSRMatrixMatVectorNumaThread(void* args);
void* CSCMatrixMatVectorNumaThread(void* args);
void* ELLMatrixMatVectorNumaThread(void* args);
void* DIAMatrixMatVectorNumaThread(void* args);

#endif