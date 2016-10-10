#include "svs.h"

double svs_sigv2_integrand(int n, double* args) {
	double r = args[0];
	return rho_nfw(r)*menc_nfw(r)/(r*r);
}

// NFW density profile
double rho_nfw(double r) {
	double x = r/vsp->Rs;
	return vsp->rho0/(x*(1.0+x)*(1.0+x));
}

double menc_nfw(double r) {
	return FOUR_PI*vsp->rho0*vsp->Rs*vsp->Rs*vsp->Rs
			*(log((vsp->Rs+r)/vsp->Rs)-r/(vsp->Rs+r));
}

// NFW density profile, Cartesian coordinates
double rho_nfw_cart(double* pos) {
	return rho_nfw(sqrt(pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2])); 
}

// velocity dispersion along the LOS given cartesian coordinates
double sigv_nfw_cart(double* pos, double* los) {

	int i;
	double sigv, mu, tmp; 
	double y0, y1, y2, y3, mu2, a0, a1, a2, a3;;
	double r = sqrt(pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2]);
	
	// interpolation of the tabulated sigma(r)
	if(r < vsp->Rvir) { 
		tmp = r/vsp->dr;
		i = floor(tmp);
		mu = tmp - i;

		// catmull-rom interpolation
		if(i > 0) y0 = vsp->sigv[i-1];
		else y0 = -vsp->sigv[1];
		y1 = vsp->sigv[i];
		y2 = vsp->sigv[i+1];
		if(i < NPOINT_SIGMA-1) y3 = vsp->sigv[i+2];
		else y3 = -vsp->sigv[NPOINT_SIGMA-1];
		mu2 = mu*mu;
		a0 = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3;
		a1 = y0 - 2.5*y1 + 2*y2 - 0.5*y3;
		a2 = -0.5*y0 + 0.5*y2;
		a3 = y1;
		sigv = (a0*mu*mu2+a1*mu2+a2*mu+a3);
	}
	else sigv = 0.0;
	return sigv;
}

// transform from the polar FOV coordinates to cartesian coordinates
void transform_to_cartesian(double* fovc, double* cart, double* los) {

	// get polar coordinates
	// compute sines and cosines
	int i;
	double r = fovc[0];
	double theta = fovc[1];
	double phi = fovc[2];
	double sl = vsp->sinl;
	double cl = vsp->cosl; 
	double sb = vsp->sinb;
	double cb = vsp->cosb; 
	double st = sin(theta);
	double ct = cos(theta);
	double sp = sin(phi);
	double cp = cos(phi);

	// get the unit vector on the sky
	los[0] = (cl*cp*sb + sl*sp)*st - cb*cl*ct;
	los[1] = cb*ct*sl + (cl*sp - cp*sb*sl)*st;
	los[2] = ct*sb + cb*cp*st;

	// scale by r, shift to the sun's position
	for(i = 0; i < 3; ++i) cart[i] = r*los[i]+vsp->pos_sun[i];
}

// NFW-weighted density profile along the LOS for the current parameters
double svs_nfw_polarproj_rho(int n, double* args) {

	double result;
	double cart[3], los[3];
	transform_to_cartesian(args, cart, los);
	result = rho_nfw_cart(cart);

	// weight by the solid angle element
	result *= sin(args[1]);
	return result;
}

double svs_nfw_polarproj_vel(int n, double* args) {

	double result, vlos;
	double cart[3], los[3];
	transform_to_cartesian(args, cart, los);
	result = rho_nfw_cart(cart);

	// LOS velocity
	vlos = -(los[0]*vsp->vel_sun[0]
		+los[1]*vsp->vel_sun[1]+los[2]*vsp->vel_sun[2]);
	result *= vlos;

	// weight by the solid angle element
	result *= sin(args[1]);
	return result;
}

double svs_nfw_polarproj_sigv(int n, double* args) {

	double result, vlos, siglos, vavg;
	double cart[3], los[3];
	transform_to_cartesian(args, cart, los);
	result = rho_nfw_cart(cart);

	// mean LOS velocity (already computed)
	// velocity dispersion along the LOS
	vavg = args[3];
	vlos = -(los[0]*vsp->vel_sun[0]
		+los[1]*vsp->vel_sun[1]+los[2]*vsp->vel_sun[2]);
	siglos = sigv_nfw_cart(cart, los);

	// weight by the mean (relative to the line centroid) and dispersion
	result *= (vlos-vavg)*(vlos-vavg)+siglos*siglos;

	// weight by the solid angle element
	result *= sin(args[1]);
	return result;
}


