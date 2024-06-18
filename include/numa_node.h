#ifndef NUMA_NODE_H
#define NUMA_NODE_H

class NumaNode4COO
{
public:
    int     alloc;
    int     core_ind;
    int     nnz;
    int     start_row;
    int     rows_per_node;
    int*    sub_row_ind;
    int*    sub_col_ind;
    double* sub_values;
    double* X;
    double* Y;
};

class NumaNode4CSR
{
public:
    int     alloc;
    int     nnz;
    int     core_ind;
    int     start_row;
    int     rows_per_node;
    int*    sub_row_ptr;
    int*    sub_col_ind;
    double* sub_values;
    double* X;
    double* Y;
};

class NumaNode4CSC
{
};

class NumaNode4ELL
{
public:
    int     nthreads;
    int     alloc;
    int     rows_per_node;
    int     nonzeros_in_row;
    int*    sub_col_ind;
    double* sub_values;
    double* X;
    double* Y;
};

class NumaNode4DIA
{
};

#endif