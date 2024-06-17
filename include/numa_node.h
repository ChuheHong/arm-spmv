#ifndef NUMA_NODE_H
#define NUMA_NODE_H

class NumaNode4ELL
{
public:
    int     nthreads;
    int     alloc;
    int     M;
    int     nonzeros_in_row;
    int*    sub_col_ind;
    double* sub_value;
    double* X;
    double* Y;
};

#endif