// C wrappers for the f90 healpix API

#include "svs.h"
#include "GNU_dv.h"

extern int __pix_tools_MOD_nside2npix(int* nside);
extern void __pix_tools_MOD_vec2pix_ring(int* nside, dope_vec_GNU* vec, int* ipring);
extern void __pix_tools_MOD_query_disc(int* nside, dope_vec_GNU* vec, double* rad, dope_vec_GNU* pixinds, int* ninds, int* nest, int* inclusive);
extern void __pix_tools_MOD_surface_triangle(dope_vec_GNU* v1, dope_vec_GNU* v2, dope_vec_GNU* v3, double* area);
extern void __pix_tools_MOD_pix2vec_ring(int* nside, int* ip, dope_vec_GNU* cent, dope_vec_GNU* vert);

int nside2npix(int nside) {
	return __pix_tools_MOD_nside2npix(&nside);
}

void vec2pix_ring(int nside, double* vec, int* ipring) {
	dope_vec_GNU dv;
	dv.base_addr = vec;
	dv.base = 0;
	dv.dtype = (1 & GFC_DTYPE_RANK_MASK) |
		((GFC_DTYPE_REAL << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK) |
		(sizeof(double) << GFC_DTYPE_SIZE_SHIFT);
	dv.dim[0].stride_mult = 1;
	dv.dim[0].lower_bound = 0;
	dv.dim[0].upper_bound = 3;
	__pix_tools_MOD_vec2pix_ring(&nside, &dv, ipring);
}

void query_disc(int nside, double* vec, double rad, int* pixinds, int* ninds, int nest, int inclusive) {
	dope_vec_GNU dv, di;
	dv.base_addr = vec;
	dv.base = 0;
	dv.dtype = (1 & GFC_DTYPE_RANK_MASK) |
		((GFC_DTYPE_REAL << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK) |
		(sizeof(double) << GFC_DTYPE_SIZE_SHIFT);
	dv.dim[0].stride_mult = 1;
	dv.dim[0].lower_bound = 0;
	dv.dim[0].upper_bound = 3;
	di.base_addr = pixinds;
	di.base = 0;
	di.dtype = (1 & GFC_DTYPE_RANK_MASK) |
		((GFC_DTYPE_INTEGER << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK) |
		(sizeof(double) << GFC_DTYPE_SIZE_SHIFT);
	di.dim[0].stride_mult = 1;
	di.dim[0].lower_bound = 0;
	di.dim[0].upper_bound = 1000000000;
	__pix_tools_MOD_query_disc(&nside, &dv, &rad, &di, ninds, &nest, &inclusive);
}

double surface_triangle(double* v1, double* v2, double* v3) {
	int i;
	double area;

#if 0
	dope_vec_GNU dv[3];
	dv[0].base_addr = v1;
	dv[1].base_addr = v2;
	dv[2].base_addr = v3;
	for(i = 0; i < 3; ++i) {
		printf("dv[%d] = %f %f %f\n", i, ((double*)dv[i].base_addr)[0], ((double*)dv[i].base_addr)[1], ((double*)dv[i].base_addr)[2]);
		dv[i].base = 0;
		dv[i].dtype = (1 & GFC_DTYPE_RANK_MASK) |
		    ((GFC_DTYPE_REAL << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK) |
		    (sizeof(double) << GFC_DTYPE_SIZE_SHIFT);
		dv[i].dim[0].stride_mult = 1;
		dv[i].dim[0].lower_bound = 0;
		dv[i].dim[0].upper_bound = 3;
	}
	__pix_tools_MOD_surface_triangle(&dv[0], &dv[1], &dv[2], &area);
#else
	double det, div;
	// Solid angle of a triangle by Oosterom and Strackee
	// Assumes v1, v2, v3 are unit vectors
	det = v1[0]*(v2[1]*v3[2]-v3[1]*v2[2])-v1[1]*(v2[0]*v3[2]-v3[0]*v2[2])+v1[2]*(v2[0]*v3[1]-v3[0]*v2[1]);
	div = 1.0;
	for(i = 0; i < 3; ++i) {
		div += v1[i]*v2[i];
		div += v2[i]*v3[i];
		div += v3[i]*v1[i];
	}
	area = 2.0*atan2(det, div);
#endif
	return area;
}

void pix2vec_ring(int nside, int ipx, double* cent, double* vert) {

	dope_vec_GNU centdesc;
	dope_vec_GNU vertdesc;
	centdesc.base_addr = cent;
	centdesc.base = 0;
	centdesc.dtype = (1 & GFC_DTYPE_RANK_MASK) |
		((GFC_DTYPE_REAL << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK) |
		(sizeof(double) << GFC_DTYPE_SIZE_SHIFT);
	centdesc.dim[0].stride_mult = 1;
	centdesc.dim[0].lower_bound = 0;
	centdesc.dim[0].upper_bound = 3;
	vertdesc.base_addr = vert;
	vertdesc.base = 0;
	vertdesc.dtype = (2 & GFC_DTYPE_RANK_MASK) |
		((GFC_DTYPE_REAL << GFC_DTYPE_TYPE_SHIFT) & GFC_DTYPE_TYPE_MASK) |
		(sizeof(double) << GFC_DTYPE_SIZE_SHIFT);
	vertdesc.dim[0].stride_mult = 1;
	vertdesc.dim[0].lower_bound = 0;
	vertdesc.dim[0].upper_bound = 3;
	vertdesc.dim[1].stride_mult = 3;
	vertdesc.dim[1].lower_bound = 0;
	vertdesc.dim[1].upper_bound = 4;
	__pix_tools_MOD_pix2vec_ring(&nside, &ipx, &centdesc, &vertdesc); 
}
