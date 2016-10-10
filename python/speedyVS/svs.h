#ifndef _SPEEDY_VS_
#define _SPEEDY_VS_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "nelder_mead.h"

#define PI_OVER_THREE 1.0471975512
#define PI 3.14159265359 
#define TWO_PI 6.28318530718 
#define SQRT_TWO_PI 2.50662827463 
#define FOUR_PI 12.5663706144
#define ONE_OVER_FOUR_PI 0.07957747154 

#define TO_RADIANS 0.01745329251

#define NPOINT_SIGMA 2048 


// struct of constants to be set from Python code
typedef struct {

	// geometry of the sun relative to the halo center
	double pos_sun[3], vel_sun[3];

	// instrumental parameters
	double theta_fov, sigma_instr, 
		   energy_range[2], bgcounts, prefac; 

	// dark matter halo data
	double Rs, rho0, Rvir; 
	double *pos, *vel, mass;
	int npart;

	// data contained here
	double* data;
	int ndata;

	// other stuff
	double b, l; // line-of-sight angles
	double cosb, cosl, sinb, sinl; // precomputed geometry for speed
	double sigv[NPOINT_SIGMA+1], dr; // velocity dispersion table 

} svs_params;


// forward declarations
double rho_nfw(double r);
double menc_nfw(double r);
double rho_nfw_cart(double* pos);
double rho_nfw_los(double s);
double sigv_nfw_cart(double* pos, double* los);
void transform_to_cartesian(double* fovc, double* cart, double* los);

// healpix wrapper declarations
int nside2npix(int nside); 
void vec2pix_ring(int nside, double* vec, int* ipring);
void query_disc(int nside, double* vec, double rad, 
		int* pixinds, int* ninds, int nest, int inclusive);
double surface_triangle(double* v1, double* v2, double* v3);
void pix2vec_ring(int nside, int ipx, double* cent, double* vert);


#endif // _SPEEDY_VS_
