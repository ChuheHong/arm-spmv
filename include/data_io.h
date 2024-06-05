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



void vec_read(const char* filename, Vector& x);
void vec_write(const char* filename, const Vector& x);

void coo_read(const char* filename, COO_Matrix& A);
void csr_read(const char* filename, CSR_Matrix& A);
void csc_read(const char* filename, CSC_Matrix& A);
void ell_read(const char* filename, ELL_Matrix& A);

#endif