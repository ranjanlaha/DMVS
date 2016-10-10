#include "svs.h"

// seeds the random number generator
void svs_random_seed(long seed) {
	if(seed) srand(seed);
	else srand(time(NULL));
}

// uniform between 1 and 0
double uniform_rvs() {
	return ((double)rand())/RAND_MAX;
}

// uses a Box-Muller transform to get two normally distributed numbers
// from two uniformly distributed ones. We throw one away here.
double normal_rvs() {
	double u1 = uniform_rvs();
	double u2 = uniform_rvs();
	return sqrt(-2.0*log(u1))*cos(TWO_PI*u2);
}

// Poisson-distributed random number generator
// from Knuth, good for small lambda
int poisson_rvs(double lam) {
	int k = 0;
	double l = exp(-lam);
	double p = 1.0;
	do {
		k++;
		p *= uniform_rvs();	
	} while(p > l);
	return k-1;
}

// generate a uniformly distributed rotation a la 
// Arno 1992, "Fast Random Rotation Matrices" 
void svs_random_rotation(double* mat) {
	double x1 = uniform_rvs(); 
	double x2 = uniform_rvs();
	double x3 = uniform_rvs(); 
	double c1 = cos(2*PI*x1);
	double s1 = sin(2*PI*x1);
	double sq3 = sqrt(x3);
	double c2r3 = cos(2*PI*x2)*sq3;
	double s2r3 = sin(2*PI*x2)*sq3;
	double sq1m3 = sqrt(1.0-x3);
	mat[3*0+0] = -(c1*(1 - 2*c2r3*c2r3)) - 2*c2r3*s1*s2r3;
	mat[3*1+0] = -((1 - 2*c2r3*c2r3)*s1) + 2*c1*c2r3*s2r3;
	mat[3*2+0] = 2*c2r3*sq1m3;
	mat[3*0+1] = 2*c1*c2r3*s2r3 + s1*(1 - 2*s2r3*s2r3);
	mat[3*1+1] = 2*c2r3*s1*s2r3 - c1*(1 - 2*s2r3*s2r3);
	mat[3*2+1] = 2*s2r3*sq1m3;
	mat[3*0+2] = 2*c1*c2r3*sq1m3 - 2*s1*s2r3*sq1m3;
	mat[3*1+2] = 2*c2r3*s1*sq1m3 + 2*c1*s2r3*sq1m3;
	mat[3*2+2] = -1 + 2*sq1m3*sq1m3;
}

// rotate an array of 3-vectors in-place
void svs_rotate_in_place(double* vec, int nvec, double* mat) {
	int i, j, v;
	double tmp[3];
	for(v = 0; v < nvec; ++v) {
		for(i = 0; i < 3; ++i) tmp[i] = vec[3*v+i];
		for(i = 0; i < 3; ++i) {
			vec[3*v+i] = 0.0;
			for(j = 0; j < 3; ++j)
				vec[3*v+i] += tmp[j]*mat[3*i+j]; 
		}
	}
}

void svs_nbody_photons(svs_params* vsp, double* sampout, int* nsamp, double fudge_ns, double fudge_nb) {

	int p, i, poiss;
	double r, vlos, mp;
	double tpos[3], tvel[3];
	double nlos[3] = {-vsp->cosl*vsp->cosb, vsp->sinl*vsp->cosb, vsp->sinb};
	double costheta = cos(vsp->theta_fov);

	// seed and sample
	srand(time(NULL));
	*nsamp = 0;
	for(p = 0; p < vsp->npart; ++p) {

		// get the distance to the particle 
		// skip if the particle is outside the fov
		for(i = 0; i < 3; ++i) {
			tpos[i] = vsp->pos[3*p+i] - vsp->pos_sun[i];
			tvel[i] = vsp->vel[3*p+i] - vsp->vel_sun[i];
		}
		r = sqrt(tpos[0]*tpos[0]+tpos[1]*tpos[1]+tpos[2]*tpos[2]);
		if(tpos[0]*nlos[0]+tpos[1]*nlos[1]+tpos[2]*nlos[2] < r*costheta) 
			continue;
	
		// Poisson-sample each individual particle by its effective rate 
		vlos = (tpos[0]*tvel[0]+tpos[1]*tvel[1]+tpos[2]*tvel[2])/(r*3.0e5);
		poiss = poisson_rvs(vsp->prefac*vsp->mass*fudge_ns/(r*r));
		for(i = 0; i < poiss; ++i) sampout[(*nsamp)++] = vlos;
	}

	// also sample some background photons into the mix
	poiss = poisson_rvs(vsp->bgcounts*fudge_nb);
	for(i = 0; i < poiss; ++i) {
		mp = uniform_rvs();
		sampout[(*nsamp)++] = vsp->energy_range[0]*mp+vsp->energy_range[1]*(1.0-mp);
	}

	// convolve the photon energies with the energy resolution
	for(i = 0; i < *nsamp; ++i)
		sampout[i] *= 1.0 + vsp->sigma_instr*normal_rvs();
}



void svs_healpix(svs_params* vsp, double* jfac, double* center, double* sigma, double* meanrad, long nside, int beam) {

	int p, i, npix, ipx, nquery;
	int* pixquery;
	double r, j, vlos, fov;
	double tpos[3], tvel[3];

	// zero the input arrays
	// bin the particles
	npix = nside2npix(nside);
	memset(jfac, 0, npix*sizeof(double));
	memset(center, 0, npix*sizeof(double));
	memset(sigma, 0, npix*sizeof(double));
	memset(meanrad, 0, npix*sizeof(double));
	if(beam) { // allocate space to sample a conical beam
		fov = TWO_PI*(1.0-cos(vsp->theta_fov));
		nquery = (int)(1.5*npix*fov*ONE_OVER_FOUR_PI);
		pixquery = malloc(nquery*sizeof(int));
	}
	else { // one pixel per particle if no beam
		pixquery = &ipx;
		nquery = 1;
	}
	for(p = 0; p < vsp->npart; ++p) {

		// get the pixel number for this particle
		for(i = 0; i < 3; ++i) {
			tpos[i] = vsp->pos[3*p+i] - vsp->pos_sun[i];
			tvel[i] = vsp->vel[3*p+i] - vsp->vel_sun[i];
		}
		r = sqrt(tpos[0]*tpos[0]+tpos[1]*tpos[1]+tpos[2]*tpos[2]);
		j = 1.0/(r*r);
		vlos = (tpos[0]*tvel[0]+tpos[1]*tvel[1]+tpos[2]*tvel[2])/(r*3.0e5);

		// get all the pixels for this particle
		if(beam) query_disc(nside, tpos, vsp->theta_fov, pixquery, &nquery, 0, 0);
		else vec2pix_ring(nside, tpos, pixquery);

		// add to the j-factor
		for(i = 0; i < nquery; ++i) {
			ipx = pixquery[i];
			jfac[ipx] += j; 
			center[ipx] += j*vlos;
			sigma[ipx] += j*vlos*vlos;
			meanrad[ipx] += j*r;
		}
	}
	for(ipx = 0; ipx < npix; ++ipx) {
		center[ipx] /= jfac[ipx];
		sigma[ipx] /= jfac[ipx];
		sigma[ipx] = sqrt(sigma[ipx]-center[ipx]*center[ipx]);
		meanrad[ipx] /= jfac[ipx];
	}
	if(beam) free(pixquery); 

#if 1
	// hack in testing of surface_triangle
	double pixest, err, a1, a2;
	double min = 1.0, max = 0.0;
	double pixar = FOUR_PI/npix;
	double cent[3], vert[4][3];
	double atot = 0.0;
	for(ipx = 0; ipx < npix; ++ipx) {
		pix2vec_ring(nside, ipx, &cent[0], &vert[0][0]);
		//a1 = surface_triangle(vert[0], vert[1], vert[2]);
		//a2 = surface_triangle(vert[2], vert[3], vert[0]);
		a1 = surface_triangle(vert[1], vert[2], vert[3]);
		a2 = surface_triangle(vert[3], vert[0], vert[1]);
		pixest = a1 + a2;
		//printf("a1 = %.5e, a2 = %.5e\n", a1, a2);
		atot += pixest;
		err = fabs(1.0-pixest/pixar);
		center[ipx] = 1.0-pixest/pixar;
		if(err < min) min = err;
		if(err > max) max = err;
		//printf("pixest[%d] = %.5e, exact = %.5e err = %f\n", ipx, pixest, pixar, err);
		//for(i = 0; i < 4; ++i)
			//printf("vert[i] = %f %f %f\n", vert[i][0], vert[i][1], vert[i][2]);
		//printf("cent = %f %f %f\n", cent[0], cent[1], cent[2]);
	}
	err = fabs(1.0-atot/FOUR_PI);
	printf("err_min = %f, err_max = %f, err_global = %.5e\n", min, max, err);
#endif
}

void svs_histogram(svs_params* vsp, double* counts, double min, double max, int nbins) {

	int p, i, bin;
	double r, vlos, invdb;
	double tpos[3], tvel[3];
	double nlos[3] = {-vsp->cosl*vsp->cosb, vsp->sinl*vsp->cosb, vsp->sinb};
	double costheta = cos(vsp->theta_fov);

	invdb = nbins/(max-min);
	memset(counts, 0, nbins*sizeof(double));
	for(p = 0; p < vsp->npart; ++p) {

		// get the distance to the particle 
		// skip if the particle is outside the fov
		for(i = 0; i < 3; ++i) {
			tpos[i] = vsp->pos[3*p+i] - vsp->pos_sun[i];
			tvel[i] = vsp->vel[3*p+i] - vsp->vel_sun[i];
		}
		r = sqrt(tpos[0]*tpos[0]+tpos[1]*tpos[1]+tpos[2]*tpos[2]);
		if(tpos[0]*nlos[0]+tpos[1]*nlos[1]+tpos[2]*nlos[2] < r*costheta) 
			continue;
	
		// find the correct energy bin in units of fractional energy shift
		// bin, weighted by the j-factor
		vlos = (tpos[0]*tvel[0]+tpos[1]*tvel[1]+tpos[2]*tvel[2])/(r*3.0e5);
		bin = floor((vlos-min)*invdb);
		if(bin < 0 || bin >= nbins) continue;
		counts[bin] += 1.0/(r*r);
	}
}


void svs_nbody(svs_params* vsp, double* jret, double* cenret, double* sigret, double* radret) {

	int p, i;
	double r, j, vlos;
	double tpos[3], tvel[3];
	double nlos[3] = {-vsp->cosl*vsp->cosb, vsp->sinl*vsp->cosb, vsp->sinb};
	double costheta = cos(vsp->theta_fov);

	double jfac = 0.0;
	double center = 0.0;
	double sigma = 0.0;
	double rad = 0.0;
	for(p = 0; p < vsp->npart; ++p) {

		// get the distance to the particle 
		// skip if the particle is outside the fov
		for(i = 0; i < 3; ++i) {
			tpos[i] = vsp->pos[3*p+i] - vsp->pos_sun[i];
			tvel[i] = vsp->vel[3*p+i] - vsp->vel_sun[i];
		}
		r = sqrt(tpos[0]*tpos[0]+tpos[1]*tpos[1]+tpos[2]*tpos[2]);
		if(tpos[0]*nlos[0]+tpos[1]*nlos[1]+tpos[2]*nlos[2] < r*costheta) 
			continue;
	
		// add to the j-factor
		j = 1.0/(r*r);
		vlos = (tpos[0]*tvel[0]+tpos[1]*tvel[1]+tpos[2]*tvel[2])/(r*3.0e5);
		jfac += j; 
		center += j*vlos;
		sigma += j*vlos*vlos;
		rad += j*r;
	}

	// return
	center /= jfac;
	sigma /= jfac;
	*jret = jfac;
	*cenret = center;
	*sigret = sqrt(sigma-center*center);
	*radret = rad/jfac;
}

