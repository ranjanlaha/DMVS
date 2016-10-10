#include "svs.h"
//#include "gsl_multimin.h"

// returns the Gaussian probability at point x, given mu and sig
double gauss_pdf(double x, double mu, double sig) {
	return exp(-(x-mu)*(x-mu)/(2.0*sig*sig))/(SQRT_TWO_PI*sig);
}

// evaluates the negative log likelihood 
// function args cast for the nelder-mead code
double gauss_plus_bg_nll(int npar, const double* par, const void* args) {

	// constrain the positive params
	svs_params* vsp = (svs_params*) args;
	double mu = par[0];
	double sig = sqrt(par[1]*par[1]+vsp->sigma_instr*vsp->sigma_instr); 
	double ns =  fabs(par[2]);
	double nb =  fabs(par[3]);
	double bg_pdf = 1.0/(vsp->energy_range[1]-vsp->energy_range[0]);
	double ll = 0.0;
	int i;
	for(i = 0; i < vsp->ndata; ++i)
		ll += log(ns*gauss_pdf(vsp->data[i], mu, sig) + nb*bg_pdf); 
	return -ll+(ns+nb); // for extended maximum likelihood
}

// evaluates the negative log likelihood 
// function args cast for the nelder-mead code
void gauss_plus_bg_invcov(svs_params* vsp, const double* par, double* icov) {

	int i, j, interv;
	double x, dx, sigfull2, gauss, unif, prob;
	double grad[4];
	double mu = par[0];
	double sig = par[1];
	double ns =  par[2];
	double nb =  par[3];

	// integration params
	int n = 4;
	int nint = 256; // TODO: use a smaller number of subintervals
	dx = (vsp->energy_range[1]-vsp->energy_range[0])/nint; 
	unif = 1.0/(vsp->energy_range[1]-vsp->energy_range[0]);
	sigfull2 = sig*sig+vsp->sigma_instr*vsp->sigma_instr;

	// midpoint rule, fine for our purposes
	memset(icov, 0, n*n*sizeof(double));
	for(interv = 0; interv < nint; ++interv) {
	
		// get the x-midpoint for this interval
		// evaluate the pdf 
		x =  vsp->energy_range[0] + (0.5+interv)*dx;
		gauss = gauss_pdf(x, mu, sqrt(sigfull2));
		prob = ns*gauss+nb*unif;

		// evaluate the gradient terms
		// apply the midpoint rule to the inverse covariance
		grad[0] = ns*gauss*(x-mu)/sigfull2;
		grad[1] = ns*gauss*sig*((x-mu)*(x-mu)-sigfull2)/(sigfull2*sigfull2);
		grad[2] = gauss;
		grad[3] = unif;
		for(i = 0; i < n; ++i)
		for(j = 0; j < n; ++j) 
			icov[n*i+j] += grad[i]*grad[j]*dx/prob;
	}
}


void svs_line_plus_bg_fit(svs_params* vsp, double* par, double* invcov) {

    int i, n, evals;      // number of dimension of initial point
	double x0[4];
    point_t    solution;
    optimset_t optimset;
    
	// set initial params
    n = 4; 
    for(i = 0; i < n; ++i) x0[i] = par[i]; 
    
	// setting default options
    optimset.tolx     = 1.0e-4; 
    optimset.tolf     = 1.0e-4;
    optimset.max_iter = 100000; 
    optimset.max_eval = 100000;
	optimset.verbose  = 0; 
    
	// call optimization methods 
    evals = nelder_mead(x0, n, optimset, &solution, gauss_plus_bg_nll, vsp);
    
	// print solution
    //printf("SOLUTION: ");
    //printf("x=[ ");
    //for(i=0; i<n; i++) {
        //printf("%.8f ", solution.x[i]);
    //}
    //printf("], fx=%.8f, nevals = %d\n", solution.fx, evals);

	// copy params back 
	// with correct sign
    par[0] = solution.x[0]; 
    for(i = 1; i < n; ++i) par[i] = fabs(solution.x[i]); 

	// calculate the covariance matrix
	gauss_plus_bg_invcov(vsp, par, invcov); 
	
}

