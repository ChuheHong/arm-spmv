#ifndef NUMA_NODE_H
#define NUMA_NODE_H

class NumaNode
{
public:
    int nthreads;
    int numanodes;
    int coreidx;
    int alloc;

    int      M;
    int      nonzeros_in_row;
    double*  value;
    double** X;
    double** Y;
};

#endif