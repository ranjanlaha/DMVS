#include <stdio.h>
#include <math.h>
#include "chealpix.h"

#define PI_OVER_THREE 1.0471975512
#define FOUR_PI 12.5663706144
#define ONE_OVER_FOUR_PI 0.07957747154 
#define STACKSZ 256

// struct of constants
typedef struct {
	
	// geometry of the sun relative to the halo center
	// and NFW profile parameters
	double rsun, vsun;
	double Rs, rho0;

	// line-of-sight parameters
	// field-of-view parameters
	double b, l;
	double theta;
	
	// geometry to avoid recomputation
	double rsun2, cosb, cosl;
	double rmax, omega;

} vs_params;


// NFW density profile
double rho_nfw(double r, vs_params* vsp) {
	double x = r/vsp->Rs;
	return vsp->rho0/(x*(1.0+x)*(1.0+x));
}

// NFW density profile, Cartesian coordinates
double rho_nfw_cart(double* pos, vs_params* vsp) {
	double x = sqrt(pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2])/vsp->Rs;
	return vsp->rho0/(x*(1.0+x)*(1.0+x));
}

// NFW density at location along an oblique LOS from the sun 
double rho_nfw_los(double s, vs_params* vsp) {
	double x = sqrt(s*s + vsp->rsun2 - 2*s*vsp->rsun*vsp->cosb*vsp->cosl)/vsp->Rs;
	return vsp->rho0/(x*(1.0+x)*(1.0+x));
}

void transform_to_cartesian(double* fovc, double* cart, vs_params* vsp) {

	int i;
	double r = fovc[0];
	double theta = fovc[1];
	double phi = fovc[2];

	// TODO: make this more efficient by precalculating sin and cos in vsp?
	double sl = sin(vsp->l);
	double cl = vsp->cosl; //cos(vsp->l);
	double sb = sin(vsp->b);
	double cb = vsp->cosb; //cos(vsp->b);
	double st = sin(theta);
	double ct = cos(theta);
	double sp = sin(phi);
	double cp = cos(phi);

	// unit vector on the sky
	cart[0] = (cl*cp*sb + sl*sp)*st - cb*cl*ct;
	cart[1] = cb*ct*sl + (cl*sp - cp*sb*sl)*st;
	cart[2] = ct*sb + cb*cp*st;

	// scale by r
	// shift to the sun's position
	for(i = 0; i < 3; ++i) cart[i] *= r;
	cart[0] += vsp->rsun;

}


// Simple adaptive quadrature based on the trapezoidal rule
// in 3D, over dR, dOmega
// Should work fine for smooth functions like the Gaussian, NFW, etc
double aq_fov(vs_params* vsp) {

	// vars
	int i, j, k, v, ax, spax;
	double q0, result, dv, err, maxerr;
	double q1[3]; // one trial integral for each split axis
		double mid[3];
	double polarcoord[3], cartcoord[3];

	// stack-based implementation
	typedef struct {
		double bounds[2][3]; // r, theta, phi
		int depth;
	} aqfov_stack; 
	aqfov_stack stack[STACKSZ];
	aqfov_stack icur, inew;
	int nstack;

	// TODO: un-hardcode these 
	double (*func)(double*, vs_params*);
	func = rho_nfw_cart;
	
	double dom = 6.28318530718*(1.0-cos(vsp->theta))*vsp->rmax;
	double tol = 1.0e-9*dom;

	double errbound = 0.0;
	int ncall = 0;

	// initialize the stack as the FOV divided in 6 wedges 
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

	// recurse on the result until the tolerance is reached
	result = 0.0;
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
			polarcoord[0] = icur.bounds[i][0];
			polarcoord[1] = icur.bounds[j][1];
			polarcoord[2] = icur.bounds[k][2];
			transform_to_cartesian(polarcoord, cartcoord, vsp);
			// Weight by sin(theta) to preserve the solid angle element
			// TODO: precalculate this for a given j?
			q0 += func(cartcoord, vsp)*sin(polarcoord[1]);
			//q0 += sin(polarcoord[1]);
			++ncall;
		}
		q0 *= 0.125*dv; 

		// evaluate the function at the midpoints
		for(ax = 0; ax < 3; ++ax) {
			q1[ax] = 0.0;
			for(i = 0; i < 2; ++i)
			for(j = 0; j < 2; ++j) {
				polarcoord[ax] = mid[ax];
				polarcoord[(ax+1)%3] = icur.bounds[i][(ax+1)%3];
				polarcoord[(ax+2)%3] = icur.bounds[j][(ax+2)%3];
			// Weight by sin(theta) to preserve the solid angle element
			// TODO: precalculate this for a given j?
				transform_to_cartesian(polarcoord, cartcoord, vsp);
				q1[ax] += func(cartcoord, vsp)*sin(polarcoord[1]);
				//q1[ax] += sin(polarcoord[1]);
				++ncall;
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
			result += q1[spax];	
			errbound += maxerr; 
			if(nstack >= STACKSZ-2) printf("Stack overflow!\n");
			continue;
		}
		for(i = 0; i < 2; ++i) {
			inew = icur;
			inew.bounds[1-i][spax] = mid[spax];
			inew.depth++;
			stack[nstack++] = inew;	
		}
	}

	result /= dom;
	errbound /= dom;

	printf(" called func %d times\n", ncall);
	printf(" result = %f, error bound = %.5e\n", result, errbound);

	// mash in all the appropriate unit conversions
	double m_s = 7.1; // sterile neutrino mass in kev
	double exposure = 300.0; // exposure time in sec
	double aperture = 1.0; // cm^2
	double gamma = 1.7428856e-28; // s^-1	
	//result /= FOUR_PI*m_s; // solid angle and sterile neutrino mass
	//result *= gamma*exposure*aperture;
	//result *= 3.086e21; // kpc to cm for the line integral 

	return result; 
}


// Simple adaptive quadrature based on the trapezoidal rule
// Should work fine for smooth functions like the Gaussian, NFW, etc
double adaptive_quadrature(vs_params* vsp) {

	// stack-based implementation
	typedef struct {
		double a, b, fa, fb;	
		int depth;
	} aq1d_stack; 
	aq1d_stack stack[STACKSZ];
	aq1d_stack icur, i1, i2;
	int nstack, ncall;
	double q1, q2, mid, fmid, result, err, errbound;

	// TODO: un-hardcode these 
	double (*func)(double, vs_params*);
	func = rho_nfw_los;
	double tol = 1.0e-8;
	printf("tol = %.5e\n", tol);

	// initialize the stack with the endpoints and recurse
	nstack = 0;
	ncall = 0;
	result = 0.0;
	errbound = 0.0;
	icur.a = 0.0;
	icur.b = vsp->rmax;
	icur.fa = func(0.0, vsp); ncall++;
	icur.fb = func(vsp->rmax, vsp); ncall++;
	icur.depth = 0;
	stack[nstack++] = icur;
	while(nstack) {
	
		// pop the stack
		icur = stack[--nstack];
	
		// get the trial integrals and relative error
		mid = 0.5*(icur.a + icur.b);
		fmid = func(mid, vsp); ncall++;
		q1 = 0.5*(icur.fa + icur.fb)*(icur.b - icur.a); // trapezoidal rule
		q2 = 0.25*(icur.fa + 2.0*fmid + icur.fb)*(icur.b - icur.a); // trapezoidal rule with two subintervals

		// update the result if the error term is small enough
		// or if the max depth has been exceeded
		err = fabs(q1-q2);
		if(err < 3.0*(icur.b-icur.a)*tol || nstack >= STACKSZ-2) {
			result += q2;	
			errbound += err;
			if(nstack >= STACKSZ-2) printf("Stack overflow!\n");
			continue;
		}
	
		// otherwise push the two subintervals to the stack 
		i1.a = icur.a; i1.b = mid;
		i1.fa = icur.fa; i1.fb = fmid;
		i1.depth = icur.depth+1;
		i2.a = mid; i2.b = icur.b;
		i2.fa = fmid; i2.fb = icur.fb;
		i2.depth = icur.depth+1;
		stack[nstack++] = i1;
		stack[nstack++] = i2;
	}

	printf(" called func %d times\n", ncall);
	printf(" error bound = %.5e\n", errbound);

	return result; 
}





// HealPix routines
#if 0

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




//////// PYTHON HELPERS ////////////////

static vs_params py_vsp;

void py_set_default_params() {
	
 	// geometry of the halo and NFW params 
		py_vsp.rsun = 8.0; //kpc 
		py_vsp.vsun = 220.0; // km/s
		py_vsp.Rs = 19.78; // kpc

		// assumed to be in M_sun/kpc^3
		// convert to keV/cm^3
		py_vsp.rho0 = 6.912562*3.7966e-5;
	
		// LOS and FOV params
	#define PI 3.14159265359 
		double torad = PI/180.0;
		py_vsp.b = 25.0*torad;
		py_vsp.l = 25.0*torad;
		py_vsp.theta = 20.0*torad;

		// compute secondary geometric stuff
		py_vsp.rsun2 = py_vsp.rsun*py_vsp.rsun;
		py_vsp.cosb = cos(py_vsp.b);
		py_vsp.cosl = cos(py_vsp.l);
		py_vsp.rmax = 10000.0;// #200.0 # TODO: check if this is large enough
		py_vsp.omega = 0.0;





}

double py_integral(int n, double* args) {

	//printf("py_vsp.cosl = %f\n", py_vsp.cosl);
	
	double rho;
	double cartcoord[3];

	transform_to_cartesian(args, cartcoord, &py_vsp);
	rho = rho_nfw_cart(cartcoord, &py_vsp);
	//rho = 1.0;

	//printf("rho = %.5e\n", rho);
	//printf("theta = %.5e, sin(theta) = %.5e\n", args[1], sin(args[1]));

	// weight for polar coordinates
	rho *= sin(args[1]);

	return rho;
}





