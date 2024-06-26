#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

class COOMatrix
{
public:
    int nrow;
    int ncol;
    int nnz;

    int*    row_ind;
    int*    col_ind;
    double* values;

    COOMatrix();
    COOMatrix(int n, int m, int nnz, int* row_ind, int* col_ind, double* values);
    COOMatrix(const COOMatrix& A);
    ~COOMatrix();
    COOMatrix& operator=(const COOMatrix& A);

    void Free();
};

class CSRMatrix
{
public:
    int nrow;
    int ncol;

    int*    row_ptr;
    int*    col_ind;
    double* values;
    double* diagonal;  // for SymGS

    CSRMatrix();
    CSRMatrix(int n, int m, int* row_ptr, int* col_ind, double* values, double* diagonal);
    CSRMatrix(const CSRMatrix& A);
    CSRMatrix(const COOMatrix& A);
    ~CSRMatrix();
    CSRMatrix& operator=(const CSRMatrix& A);
    CSRMatrix& operator=(const COOMatrix& A);

    void Free();
};

class CSCMatrix
{
public:
    int nrow;
    int ncol;

    int*    row_ind;
    int*    col_ptr;
    double* values;

    CSCMatrix();
    CSCMatrix(int n, int m, int* row_ind, int* col_ptr, double* values);
    CSCMatrix(const CSCMatrix& A);
    CSCMatrix(const COOMatrix& A);
    ~CSCMatrix();
    CSCMatrix& operator=(const CSCMatrix& A);
    CSCMatrix& operator=(const COOMatrix& A);

    void Free();
};

// 这里的ELL是列主序存储
class ELLMatrix
{
public:
    int nrow;
    int ncol;
    int nnz;
    int nonzeros_in_row;

    int*    col_ind;
    double* values;
    double* diagonal;  // for SymGS

    ELLMatrix();
    ELLMatrix(int n, int m, int nnz, int nonzeros_in_row, int* col_ind, double* values, double* diagonal);
    ELLMatrix(const ELLMatrix& A);
    ELLMatrix(const COOMatrix& A);
    ~ELLMatrix();
    ELLMatrix& operator=(const ELLMatrix& A);
    ELLMatrix& operator=(const COOMatrix& A);

    void Free();
};

class BlockMatrix
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

    BlockMatrix();
    BlockMatrix(int n, int m, int nnz, int nblocks, int* block_size, int* row_ind, int* col_ind, double** values);
    BlockMatrix(const BlockMatrix& A);
    BlockMatrix(const COOMatrix& A);
    ~BlockMatrix();
    BlockMatrix& operator=(const BlockMatrix& A);
    BlockMatrix& operator=(const COOMatrix& A);

    void Free();
};

class DIAMatrix
{
public:
    int nnz;
    int nrow;
    int ncol;
    int ndiags;  // Number of diagonals

    int*    offsets;  // Array of diagonal offsets
    double* values;   // Array of values for the diagonals

    DIAMatrix();
    DIAMatrix(int n, int m, int ndiags, int* offsets, double* values);
    DIAMatrix(const DIAMatrix& A);
    DIAMatrix(const CSRMatrix& A);
    ~DIAMatrix();
    DIAMatrix& operator=(const DIAMatrix& A);
    DIAMatrix& operator=(const CSRMatrix& A);

    void Free();
};

#endif