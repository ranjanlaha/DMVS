#!/lustre/ki/pfs/dmpowel1/anaconda/bin/python -u

from scipy.stats import norm, chi2
import numpy as np
import sys
sys.path.insert(0, '/afs/slac.stanford.edu/u/gl/dmpowel1/DMVS/python')
import classyVS as vs

# constants
mx_lon = np.array([-105., -65., -25., 25., 65., 105.])
mx_lat = 25.

nsamp = 1024 #30 
npoint = mx_lon.size

# runs sampling on all haloes! 
if __name__ == '__main__':

	# load the proper halo
	# fudge the number of photons
	h = int(sys.argv[1])

	### Set all parameters, use Micro-X defaults ###
	mxvs = vs.velocitySpectroscopy(halo=h, verbose=True)
	fracs = np.array([0.01, 0.05, 0.1, 0.25, 0.5])
	numct = np.zeros((len(mx_lon), len(fracs)), dtype=np.int32)

	# get parameters and covariances for all samples and pointings
	for i in xrange(nsamp):

	    # randomly orient the halo
		# MC sample photons directly from the N-body particles
		mxvs.rotateHaloInPlace()
		for l, lon in enumerate(mx_lon):

			# get the fraction of pointings that have a single subhalo
			# contributing more than a certain percentage of the total flux
			hcts, truects, hinds = mxvs.doHaloes(lon=lon, lat=mx_lat, minpart=100)
			fluxfrac = hcts/truects

			for p, frac in enumerate(fracs):
				numct[l][p] += np.any(fluxfrac > frac)
																		        
		if i%10 == 0:
			print " sample %d of %d" % (i, nsamp)

	print "fractions = ", fracs 
	print "percentiles = ", numct 
	print "nsamp = ", nsamp

	fname = 'halo%d.%s' % (h, 'halofrac')
	with open(fname, 'wb') as f:
		np.save(f, numct)

