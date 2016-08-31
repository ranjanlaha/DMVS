#include <stdio.h>
//#include "chealpix.h"

//extern void __pix_tools_MOD_pix2vec_ring(int* nside, int* ip, double* cent, double* vert);
extern void __pix_tools_MOD_pix2vec_ring(int*, int*, double*[3]);//, double[]);
extern int __pix_tools_MOD_nside2npix(int* nside);


int bin_pixels(double* pos, double* vel) {
	
	//printf("npix2nside(3072) = %ld.\n", npix2nside(3072));

	int nside = 16;
	int ip = 1532;
	//struct {
		//double x, y, z;
	//} cent;
	double cent[3];
	//double vert[3*4];
	printf("nside2npix(16) = %ld.\n", __pix_tools_MOD_nside2npix(&nside));
	__pix_tools_MOD_pix2vec_ring(&nside, &ip, &cent);//, vert);

	//int i;
	//for(i = 0; i < 4; ++i)
		//printf("vert[i] = %f %f %f\n", vert[3*i+0], vert[3*i+1], vert[3*i+2]);


	return 0;
}
