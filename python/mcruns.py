#!/lustre/ki/pfs/dmpowel1/anaconda/bin/python -u

from scipy.stats import norm, chi2
import numpy as np
import sys
sys.path.insert(0, '/afs/slac.stanford.edu/u/gl/dmpowel1/DMVS/python')
import classyVS as vs

# constants
mx_lon = np.array([-105., -65., -25., 25., 65., 105.])
mx_lat = 25.
torad = np.pi/180.0
decosb = -220./3.0e5*np.cos(mx_lat*torad)

nsamp = 4096
npoint = mx_lon.size

# runs sampling on all haloes! 
if __name__ == '__main__':

	# load the proper halo
	# fudge the number of photons
	h = int(sys.argv[1])
	fud = int(sys.argv[2])

	### Set all parameters, use Micro-X defaults ###
	mxvs = vs.velocitySpectroscopy(halo=h, verbose=True)
	print ' photon count fudge factor =', fud

	# get parameters and covariances for all samples and pointings
	allpar = np.empty((nsamp, npoint, 4))
	allcov = np.empty((nsamp, npoint, 4, 4))
	for i in xrange(nsamp):

	    # randomly orient the halo
		# MC sample photons directly from the N-body particles
		mxvs.rotateHaloInPlace()
		for l, lon in enumerate(mx_lon):
						        
			# extended maximum-likelihood fit on the photons
			phot = mxvs.samplePhotonsFromNBody(lon=lon, lat=25.0, fudge=fud)
			par = np.array([decosb*np.sin(lon*torad), 5.0e-4, 5.0*fud, 1.0*fud]) # initial guess
			allpar[i][l], allcov[i][l] = mxvs.fitLinePlusBG(phot, par0=par)
																		        
		if i%10 == 0:
			print " sample %d of %d" % (i, nsamp)

	fname = 'halo%d.%d.%s' % (h, fud, 'fit')
	ndata = np.array([nsamp], dtype=np.int32)
	with open(fname, 'wb') as f:
		ndata.tofile(f)
		allpar.tofile(f)
		allcov.tofile(f)
		ndata.tofile(f)
		f.close()

