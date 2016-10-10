#ifndef _NELDER_MEAD_H_
#define _NELDER_MEAD_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// parameters for the algorithm
#define RHO      1
#define CHI      2
#define GAMMA    0.5
#define SIGMA    0.5

// define the point in the simplex (x) and its functional value (fx)
typedef struct{
    double *x;
    double fx;
} point_t;

// define optimization settings
typedef struct{
    double tolx;
    double tolf;
    int max_iter;
    int max_eval;
	int iter_count;
	int eval_count;
	int verbose;
} optimset_t;

// Nelder-Mead algorithm and template cost function 
int nelder_mead(double *x0, int n, optimset_t optimset, point_t *solution, 
	double(*fun)(int, const double*, const void*), void *fun_args);


#endif // _NELDER_MEAD_H_
