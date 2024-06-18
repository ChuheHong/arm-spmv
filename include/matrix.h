/**
 * @file    : sparse_matrix.h
 * @author  : theSparky Team
 * @version :
 *
 * Functions for Sparse Matrix Init.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

class COO_Matrix
{
public:
    int nrow;
    int ncol;
    int nnz;

    int*    row_ind;
    int*    col_ind;
    double* values;

    COO_Matrix();
    COO_Matrix(int n, int m, int nnz, int* row_ind, int* col_ind, double* values);
    COO_Matrix(const COO_Matrix& A);
    ~COO_Matrix();
    COO_Matrix& operator=(const COO_Matrix& A);

    void Free();
};

class CSR_Matrix
{
public:
    int nrow;
    int ncol;

    int*    row_ptr;
    int*    col_ind;
    double* values;
    double* diagonal;  // for SymGS

    CSR_Matrix();
    CSR_Matrix(int n, int m, int* row_ptr, int* col_ind, double* values, double* diagonal);
    CSR_Matrix(const CSR_Matrix& A);
    CSR_Matrix(const COO_Matrix& A);
    ~CSR_Matrix();
    CSR_Matrix& operator=(const CSR_Matrix& A);
    CSR_Matrix& operator=(const COO_Matrix& A);

    void Free();
};

class CSC_Matrix
{
public:
    int nrow;
    int ncol;

    int*    row_ind;
    int*    col_ptr;
    double* values;

    CSC_Matrix();
    CSC_Matrix(int n, int m, int* row_ind, int* col_ptr, double* values);
    CSC_Matrix(const CSC_Matrix& A);
    CSC_Matrix(const COO_Matrix& A);
    ~CSC_Matrix();
    CSC_Matrix& operator=(const CSC_Matrix& A);
    CSC_Matrix& operator=(const COO_Matrix& A);

    void Free();
};

// 这里的ELL是列主序存储
class ELL_Matrix
{
public:
    int nrow;
    int ncol;
    int nnz;
    int nonzeros_in_row;

    int*    col_ind;
    double* values;
    double* diagonal;  // for SymGS

    ELL_Matrix();
    ELL_Matrix(int n, int m, int nnz, int nonzeros_in_row, int* col_ind, double* values, double* diagonal);
    ELL_Matrix(const ELL_Matrix& A);
    ELL_Matrix(const COO_Matrix& A);
    ~ELL_Matrix();
    ELL_Matrix& operator=(const ELL_Matrix& A);
    ELL_Matrix& operator=(const COO_Matrix& A);

    void Free();
};

class Block_Matrix
{
public:
    int nrow;
    int ncol;
    int nnz;
    int nblocks;

    int*     block_size;
    int*     row_ind;
    int*     col_ind;
    double** values;

    Block_Matrix();
    Block_Matrix(int n, int m, int nnz, int nblocks, int* block_size, int* row_ind, int* col_ind, double** values);
    Block_Matrix(const Block_Matrix& A);
    Block_Matrix(const COO_Matrix& A);
    ~Block_Matrix();
    Block_Matrix& operator=(const Block_Matrix& A);
    Block_Matrix& operator=(const COO_Matrix& A);

    void Free();
};

class DIA_Matrix
{
public:
    int nrow;
    int ncol;
    int ndiags;  // Number of diagonals

    int*    offsets;  // Array of diagonal offsets
    double* values;   // Array of values for the diagonals

    DIA_Matrix();
    DIA_Matrix(int n, int m, int ndiags, int* offsets, double* values);
    DIA_Matrix(const DIA_Matrix& A);
    DIA_Matrix(const CSR_Matrix& A);
    ~DIA_Matrix();
    DIA_Matrix& operator=(const DIA_Matrix& A);
    DIA_Matrix& operator=(const CSR_Matrix& A);

    void Free();
};

#endif