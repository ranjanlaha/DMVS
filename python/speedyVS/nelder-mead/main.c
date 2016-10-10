#include "nelder_mead.h"

// define a dummy cost function
// cost function f : R^n->R
#define SQUARE(x) ((x)*(x))
double sum_of_squares(int n, const double *x, const void *arg) {
    int i;
    double out;
    
	// Sum Squares Function
    out = 0;
    for(i=0; i<n; i++) {
        out += SQUARE((i+1)*x[i]);
    }
    return 0.5 * out;
}

int main( int argc, const char* argv[] ) {
    int i, n;      // number of dimension of initial point
    double *x0;    // cooridnates of initial point
    
    point_t    solution;
    optimset_t optimset;
    
    if(argc==1) {
        printf("%s: error: not enough inputs \n", argv[0]);
        return 0;
    }
    
	// reading initial point from command line
    n = argc-1;
    x0 = malloc(n*sizeof(double));
    for(i=0; i<n; i++) {
        x0[i] = atof(argv[i+1]);
    }
    
	// setting default options
    optimset.tolx     = 1.0e-6;
    optimset.tolf     = 1.0e-6;
    optimset.max_iter = 10000;
    optimset.max_eval = 10000;
	optimset.verbose  = 0; 
    
	// call optimization methods 
    nelder_mead(x0, n, optimset, &solution, sum_of_squares, NULL);
    
	// print solution
    printf("SOLUTION\n");
    printf("x=[ ");
    for(i=0; i<n; i++) {
        printf("%.8f ", solution.x[i]);
    }
    printf("], fx=%.8f \n", solution.fx);
    
    return 0;
}
