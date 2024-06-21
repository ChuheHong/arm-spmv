/**
 * @file    : data_io.cpp
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Matrix and Vector Read and Write.
 */
#ifndef DATA_IO_H
#define DATA_IO_H

#include <stdio.h>

#include "matrix.h"
#include "vector.h"

void VectorRead(const char* filename, Vector& x);
void VectorWrite(const char* filename, const Vector& x);

void COOMatrixRead(const char* filename, COOMatrix& A);
void CSRMatrixRead(const char* filename, CSRMatrix& A);
void CSCMatrixRead(const char* filename, CSCMatrix& A);
void ELLMatrixRead(const char* filename, ELLMatrix& A);

#endif