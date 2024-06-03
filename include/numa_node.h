#ifndef NUMA_NODE_H
#define NUMA_NODE_H

class NumaNode
{
public:
    int      nthreads;
    int      numanodes;
    int      coreidx;
    int      alloc;
    int      M;
    int      nonzeros_in_row;
    int*    sub_col_ind;
    double* sub_value;
    double* X;
    double* Y;
};

#endif