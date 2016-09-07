#include <stdio.h>
#include <math.h>

#define PI_OVER_THREE 1.0471975512
#define PI 3.14159265359 
#define TWO_PI 6.28318530718 
#define FOUR_PI 12.5663706144
#define ONE_OVER_FOUR_PI 0.07957747154 
#define STACKSZ 256

// forward declarations
double rho_nfw(double r);
double rho_nfw_cart(double* pos);
double rho_nfw_los(double s);
double sigv_nfw_cart(double* pos, double* los);
void transform_to_cartesian(double* fovc, double* cart, double* los);

// struct of constants to be set from Python code
typedef struct {
	double rsun, vsun; // geometry of the sun relative to the halo center
	double Rs, rho0; // NFW profile parameters
	double b, l; // line-of-sight angles
	double theta, rmax; // field-of-view parameters
	double cosb, cosl, sinb, sinl; // precomputed geometry for speed
} vs_params;
static vs_params vsp;



// set the static parameters struct from Python
void py_set_params(vs_params* py_vsp) {
	vsp = *py_vsp;
}

// NFW-weighted density profile along the LOS for the current parameters
double py_nfw_polarproj_rho(int n, double* args) {

	double result;
	double cart[3], los[3];
	transform_to_cartesian(args, cart, los);
	result = rho_nfw_cart(cart);

	// weight by the solid angle element
	result *= sin(args[1]);
	return result;
}

double py_nfw_polarproj_vel(int n, double* args) {

	double result, vlos;
	double cart[3], los[3];
	transform_to_cartesian(args, cart, los);
	result = rho_nfw_cart(cart);

	// y-component of the LOS
	vlos = los[1]*vsp.vsun;
	result *= vlos;

	// weight by the solid angle element
	result *= sin(args[1]);
	return result;
}

double py_nfw_polarproj_sigv(int n, double* args) {

	double result, vlos, siglos, vavg;
	double cart[3], los[3];
	transform_to_cartesian(args, cart, los);
	result = rho_nfw_cart(cart);

	// mean LOS velocity (already computed)
	// y-component of the LOS
	// velocity dispersion along the LOS
	vavg = args[3];
	vlos = los[1]*vsp.vsun;
	siglos = sigv_nfw_cart(cart, los);

	// weight by the mean (relative to the line centroid) and dispersion
	result *= (vlos-vavg)*(vlos-vavg)+siglos*siglos;

	// weight by the solid angle element
	result *= sin(args[1]);
	return result;
}


// NFW density profile
double rho_nfw(double r) {
	double x = r/vsp.Rs;
	return vsp.rho0/(x*(1.0+x)*(1.0+x));
}

// NFW density profile, Cartesian coordinates
double rho_nfw_cart(double* pos) {
	return rho_nfw(sqrt(pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2])); 
}

// NFW density at location along an oblique LOS from the sun 
double rho_nfw_los(double s) {
	return rho_nfw(sqrt(s*s + vsp.rsun*vsp.rsun - 2*s*vsp.rsun*vsp.cosb*vsp.cosl));
}

// velocity dispersion along the LOS given cartesian coordinates
double sigv_nfw_cart(double* pos, double* los) {
	return 0.0;
}

// transform from the polar FOV coordinates to cartesian coordinates
void transform_to_cartesian(double* fovc, double* cart, double* los) {

	// get polar coordinates
	// compute sines and cosines
	double r = fovc[0];
	double theta = fovc[1];
	double phi = fovc[2];
	double sl = vsp.sinl;
	double cl = vsp.cosl; 
	double sb = vsp.sinb;
	double cb = vsp.cosb; 
	double st = sin(theta);
	double ct = cos(theta);
	double sp = sin(phi);
	double cp = cos(phi);

	// get the unit vector on the sky
	los[0] = (cl*cp*sb + sl*sp)*st - cb*cl*ct;
	los[1] = cb*ct*sl + (cl*sp - cp*sb*sl)*st;
	los[2] = ct*sb + cb*cp*st;

	// scale by r, shift to the sun's position
	int i;
	for(i = 0; i < 3; ++i) cart[i] = r*los[i];
	cart[0] += vsp.rsun;
}









//////////////////////////////////
// old, deprecated stuff        //
//////////////////////////////////
#if 0

// NFW profile along the LOS for the current parameters
double py_nfw_los(double s) {
	return rho_nfw_los(s); 
}

//////// My own adaptive quadrature routines /////////

// Simple adaptive quadrature based on the trapezoidal rule
// in 3D, over dR, dOmega
// Should work fine for smooth functions like the Gaussian, NFW, etc
int aq_fov(double* result, double* errbound, int* ncall, double tolerance, vs_params* vsp) {

	// stack-based implementation
	typedef struct {
		double bounds[2][3]; // r, theta, phi
		int depth;
	} aqfov_stack; 
	aqfov_stack stack[STACKSZ];
	aqfov_stack icur, inew;
	int i, j, k, v, ax, spax, nstack;
	double q0, dv, err, maxerr;
	double q1[3], mid[3]; // one trial integral for each split axis
	double polar[3], cart[3], los[3];

	// TODO: un-hardcode these 
	double (*func)(double*, vs_params*) = rho_nfw_cart;
	double tol = tolerance/(vsp->rmax*vsp->theta*TWO_PI);

	// initialize the stack as the FOV divided in 6 wedges 
	// recurse on the result until the tolerance is reached
	*ncall = 0;
	*result = 0.0;
	*errbound = 0.0;
	nstack = 0;
	for(v = 0; v < 6; ++v) {
		inew.bounds[0][0] = 0.0;
		inew.bounds[1][0] = vsp->rmax;
		inew.bounds[0][1] = 0.0; 
		inew.bounds[1][1] = vsp->theta;
		inew.bounds[0][2] = PI_OVER_THREE*v; 
		inew.bounds[1][2] = PI_OVER_THREE*(v+1); 
		inew.depth = 0;
		stack[nstack++] = inew;
	}
	while(nstack) {
	
		// pop the stack
		icur = stack[--nstack];

		// get the midpoints of the integration subdomains
		// also the volume element
		dv = 1.0; // one-eight of the volume element
		for(ax = 0; ax < 3; ++ax) {
			mid[ax] = 0.5*(icur.bounds[0][ax]+icur.bounds[1][ax]);
			dv *= icur.bounds[1][ax]-icur.bounds[0][ax];
		}

		// evaluate the function at the corners and estimate the first integral
		q0 = 0.0;
		for(i = 0; i < 2; ++i)
		for(j = 0; j < 2; ++j)
		for(k = 0; k < 2; ++k) {
			polar[0] = icur.bounds[i][0];
			polar[1] = icur.bounds[j][1];
			polar[2] = icur.bounds[k][2];
			transform_to_cartesian(polar, cart, los, vsp);
			// Weight by sin(theta) to preserve the solid angle element
			// TODO: precalculate this for a given j?
			q0 += func(cart, vsp)*sin(polar[1]);
			//q0 += sin(polarcoord[1]);
			(*ncall)++;
		}
		q0 *= 0.125*dv; 

		// evaluate the function at the midpoints
		for(ax = 0; ax < 3; ++ax) {
			q1[ax] = 0.0;
			for(i = 0; i < 2; ++i)
			for(j = 0; j < 2; ++j) {
				polar[ax] = mid[ax];
				polar[(ax+1)%3] = icur.bounds[i][(ax+1)%3];
				polar[(ax+2)%3] = icur.bounds[j][(ax+2)%3];
				// Weight by sin(theta) to preserve the solid angle element
				// TODO: precalculate this for a given j?
				transform_to_cartesian(polar, cart, los, vsp);
				q1[ax] += func(cart, vsp)*sin(polar[1]);
				//q1[ax] += sin(polarcoord[1]);
				(*ncall)++;
			}
			// multiply by the correct volume factor
			q1[ax] *= 0.125*dv; 
			q1[ax] += 0.5*q0;
		} 

		// find the split axis with the largest error 
		// update the result if the error term is small enough
		// or if the max depth has been exceeded
		// otherwise split axes and recurse along the one with the largest error
		maxerr = 0.0;
		spax = 0;
		for(ax = 0; ax < 3; ++ax) {
			err = fabs(q0-q1[ax]);
			if(err > maxerr) {
				maxerr = err;
				spax = ax;
			}
		}
		if(maxerr < dv*tol || nstack >= STACKSZ-2) {
			*result += q1[spax];	
			*errbound += maxerr; 
			continue;
		}
		if(nstack >= STACKSZ-2) return 0; 
		for(i = 0; i < 2; ++i) {
			inew = icur;
			inew.bounds[1-i][spax] = mid[spax];
			inew.depth++;
			stack[nstack++] = inew;	
		}
	}
	return 1; 
}


// Simple adaptive quadrature based on the trapezoidal rule
// Should work fine for smooth functions like the Gaussian, NFW, etc
int aq_los(double* result, double* errbound, int* ncall, double tolerance, vs_params* vsp) {

	// stack-based implementation
	typedef struct {
		double a, b, fa, fb;	
		int depth;
	} aq1d_stack; 
	aq1d_stack stack[STACKSZ];
	aq1d_stack icur, i1, i2;
	int nstack;
	double q1, q2, mid, fmid, err;

	// TODO: un-hardcode these 
	double (*func)(double, vs_params*) = rho_nfw_los;
	double tol = tolerance/vsp->rmax;

	// initialize the stack with the endpoints and recurse
	*ncall = 0;
	*result = 0.0;
	*errbound = 0.0;
	nstack = 0;
	icur.a = 0.0;
	icur.b = vsp->rmax;
	icur.fa = func(0.0, vsp); (*ncall)++;
	icur.fb = func(vsp->rmax, vsp); (*ncall)++;
	icur.depth = 0;
	stack[nstack++] = icur;
	while(nstack) {
	
		// pop the stack
		icur = stack[--nstack];
	
		// get the trial integrals and relative error
		mid = 0.5*(icur.a + icur.b);
		fmid = func(mid, vsp); (*ncall)++;
		q1 = 0.5*(icur.fa + icur.fb)*(icur.b - icur.a); // trapezoidal rule
		q2 = 0.25*(icur.fa + 2.0*fmid + icur.fb)*(icur.b - icur.a); // trapezoidal rule with two subintervals

		// update the result if the error term is small enough
		// return error if the max depth has been exceeded
		// otherwise push the two subintervals to the stack 
		err = fabs(q1-q2);
		if(err < 3.0*(icur.b-icur.a)*tol || nstack >= STACKSZ-2) {
			*result += q2;	
			*errbound += err;
			continue;
		}
		if(nstack >= STACKSZ-2) return 0; 
		i1.a = icur.a; i1.b = mid;
		i1.fa = icur.fa; i1.fb = fmid;
		i1.depth = icur.depth+1;
		i2.a = mid; i2.b = icur.b;
		i2.fa = fmid; i2.fb = icur.fb;
		i2.depth = icur.depth+1;
		stack[nstack++] = i1;
		stack[nstack++] = i2;
	}
	return 1;
}




#endif





// HealPix routines
#if 0

//#include "chealpix.h"
int bin_pixels(double* pos, double* vel) {
	
	printf("npix2nside(3072) = %ld.\n", npix2nside(3072));

	int nside = 16;
	int ip = 1532;
	double cent[3];
	pix2vec_ring(nside, ip, cent);

	printf("Centroid[%d] = %f %f %f\n", ip, cent[0], cent[1], cent[2]);


	return 0;
}


#endif



