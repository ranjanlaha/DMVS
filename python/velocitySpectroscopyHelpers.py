import numpy as np
import healpy as hp
import scipy.integrate as integrate

# filters particle data into a cone based on latitude and longitude of the los
# along with the angular radius of the sampling cone
# default los is the fiducial observing direction from Speckhard+2016
# returns the LOS velocities and their distances
def sample_los(pos, vel, lon=20.0, lat=5.0, beam_radius=2.5):
    
    # convert from degrees to radians and make the LOS vector
    torad = np.pi/180.0
    lonr, latr, beamr = lon*torad, lat*torad, beam_radius*torad
    n_los = np.array([-np.cos(lonr)*np.cos(latr), np.sin(lonr)*np.cos(latr), np.sin(latr)])
    
    # get all particles in the LOS cone
    dist = np.sum(pos**2, axis=1)**0.5 # scalar distance to each particle
    inds = np.sum(pos*n_los, axis=1) > dist*np.cos(beamr)
    
    # return the positions, the LOS velocity, and the distance for each particle
    return pos[inds], np.sum(vel[inds]*pos[inds]/dist[inds,None], axis=1), dist[inds]

# compute the observed DM decay line centroid uncertainty for the given N-body data and
# instrumental response parameters
def spectroscopy(pos, vel, partmass, lon=[-85., -65., -45., -25., 0., 25., 45., 65., 85.], lat=25.0,
		aperture=1.0, exposure=300.0, fov=0.3789224, sigma_energy=0.0003639950, theta_sample=20.0):
	
	### Sterile neutrino parameters from Bulbul+2014 ###
	# TODO: Assumes 3.5 keV observing energy for now!!!
	en = 3.5 # keV photons
	m_x = 7.1 # mass in keV
	Gamma = (1.38e-29)*(7e-4)*(7.1)**5 # decay rate in s^-1

	# Get the number of background photons for this observation 
	background = aperture*exposure*fov*integrate_background(en, sigma_energy)
	
	# mash all model parameters into a prefactor with correct unit conversions, 
	# so that 1/r^2 weighting (r in kpc) is all that is required
	# to get the photon number count into the detector
	prefac = Gamma*exposure/(4.*np.pi) # decay rate, exposure time, solid angle
	prefac *= partmass/m_x*1.12e63 # number of sterile neutrinos per simulation particle
	prefac *= aperture*1.050265e-43 # put aperture in units of kpc^2
	prefac *= fov/(2*np.pi*(1.-np.cos(theta_sample*np.pi/180.)))  # properly scale counts by the size of the sampling cone 
	
	# sample and plot
	lonr = np.array(lon)
	cenr = np.zeros_like(lonr)
	sigr = np.zeros_like(lonr)
	ctsr = np.zeros_like(lonr)
	for i, l in enumerate(lon): 
		
		# sample along the los cone and calculate the los velocity
		# get the fractional energy shift from the speed of light
		_, v_los, r_los = sample_los(pos, vel, lon=l, lat=25.0, beam_radius=theta_sample)
		v_los /= 3.0e5 # divide by speed of light in km/s to convert to fractional energy shift
		
		# mean energy and width of the dispersed line emission
		center = np.average(v_los, weights=r_los**-2)
		sigma_line = np.average((v_los-center)**2, weights=r_los**-2)**0.5
		
		# effective line width when convolved with Astro-H's energy resolution
		sigma_eff = (sigma_line**2 + sigma_energy**2)**0.5
		
		# get the photon number count for the given exposure
		Ns = prefac*np.sum(r_los**-2)
		
		# Cramer-Rao bound for backgrounds
		C_R = cramer_rao(sig=Ns, bg=background)
		
		# the error in the centroid according to Poisson stats
		sigma_center = C_R * sigma_eff * Ns**-0.5

		# append to return values
		cenr[i] = center
		sigr[i] = sigma_center
		ctsr[i] = Ns 

		print '.', # print a little progress indicator

	print ''
	ret_dict = {'l': lonr, 'centroid': cenr, 'sigma_centroid': sigr, 'counts': ctsr}
	return ret_dict 

# compute the observed DM decay line centroid uncertainty for the given N-body data and
# instrumental response parameters
def spectroscopy_healpix(pos, vel, partmass, nside=16, sigma_energy=0.0003639950):
	
	### Sterile neutrino parameters from Bulbul+2014 ###
	# TODO: Assumes 3.5 keV observing energy for now!!!
	en = 3.5 # keV photons
	m_x = 7.1 # mass in keV
	Gamma = (1.38e-29)*(7e-4)*(7.1)**5 # decay rate in s^-1

	# Healpix binning info
	dist = np.sum(pos**2, axis=1)**0.5 # scalar distance to each particle
	de = np.sum(vel*pos, axis=1)/(dist*3.0e5) #  fractional energy shift of each particle
	hpinds = hp.vec2pix(nside, pos[:,0], pos[:,1], pos[:,2]) # healpix indices
	invd2 = dist**-2 # for 1/r^2 weighting
	minl = hp.nside2npix(nside) # size of healpix array
	
	# mash all model parameters into a prefactor with correct unit conversions
	prefac = Gamma*1.050265e-43/(4.*np.pi) # decay rate and scale area element to cm^2
	prefac *= partmass/m_x*1.12e63 # number of sterile neutrinos per simulation particle
	prefac *= minl/(4*np.pi) # divide by the solid angle per pixel  

	# Bin everything
	flux = np.bincount(hpinds, weights=invd2, minlength=minl) # total flux
	center = np.bincount(hpinds, weights=de*invd2, \
			minlength=minl)/flux # line centroid fractional shift
	sigma_line = (np.bincount(hpinds, weights=invd2*(de-center[hpinds])**2, \
			minlength=minl)/flux)**0.5 # line width
	flux *= prefac # convert flux to units of photons/cm^2/str/s 

	# the error in the centroid according to Poisson stats
	# effective line width when convolved with the energy resolution
	# Cramer-Rao bound for backgrounds
	# X-ray background in photons/cm^2/str/s 
	background = integrate_background(en, sigma_energy)
	C_R = cramer_rao(sig=flux, bg=background)
	sigma_eff = (sigma_line**2 + sigma_energy**2)**0.5
	#### TODO: make this independent of the healpix "fov" ####
	sigma_center = C_R * sigma_eff * flux**-0.5

	ret_dict = {'centroid': center, 'sigma_centroid': sigma_center, 'flux': flux}
	return ret_dict 

# radially bins particles to create a mass profile for the halo
# returns rho(r) and M_enc(r)
def mass_profile(pos, vel, partmass, rmin=0.68, rmax=300., nbins=100):
	    
	# make the bins and the volume element
	bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
	dV = 4./3*np.pi*(bins[1:]-bins[:-1])* \
	    (bins[1:]**2+bins[1:]*bins[:-1]+bins[:-1]**2) # difference of cubes for stability
	rpart = np.sum(pos**2, axis=1)**0.5
	rnorm = pos/rpart[:, None] # unit vector to each particle
	retdict = {}
	    
	# bin the particles to get rho
	# compute the bin radius as the mass-weighted mean for each bin
	num, _ = np.histogram(rpart, bins)
	radw, _ = np.histogram(rpart, bins, weights=rpart)
	rho = partmass*num/dV
	r = radw/num

	# compute the velocity dispersion profile
	# radial velocity component 
	vrad = np.sum(vel*rnorm, axis=1) 
	vr_av, _ = np.histogram(rpart, bins, weights=vrad)
	vr2_av, _ = np.histogram(rpart, bins, weights=vrad**2)
	sigv_rad = (vr2_av/num-(vr_av/num)**2)**0.5
	# tangential dispersion, assume the mean is zero
	vt2_av, _ = np.histogram(rpart, bins, \
			weights=np.sum((vel-vrad[:,None]*rnorm)**2, axis=1))
	sigv_tan = (vt2_av/num)**0.5
	# total dispersion, assume the mean is zero
	vtot2_av, _ = np.histogram(rpart, bins, \
			weights=np.sum(vel**2, axis=1))
	sigv_tot = (vtot2_av/num)**0.5
	
	# get density and filter out zero-density entries
	nfilt = num > 0
	rho = rho[nfilt]
	r = r[nfilt]
	sigv_rad = sigv_rad[nfilt]
	sigv_tan = sigv_tan[nfilt]
	sigv_tot = sigv_tot[nfilt]
	num = num[nfilt]
	retdict['r'] = r
	retdict['rho'] = rho
	retdict['sigma_rho'] = rho*num**-0.5
	retdict['vdisp_rad'] = sigv_rad 
	retdict['vdisp_tan'] = sigv_tan 
	retdict['vdisp_tot'] = sigv_tot 
	
	# re-bin particles for the enclosed mass, so that the radii for rho and m_enc are the same
	# Include r=0 as a bin edge for this
	num, _ = np.histogram(rpart, np.insert(r, 0, 0.0))
	csum = np.cumsum(num)
	menc = partmass*csum
	retdict['menc'] = menc
	retdict['sigma_menc'] = menc*csum**-0.5 
	
	# give back all profiles
	return retdict 
    

# Optimal Cramer-Rao bound on uncertainty with backgrounds
def cramer_rao(sig=1, bg=0):
    return (1.+4.*(bg/sig))**0.5

def integrate_background(energy, sigma_energy):

	# CXB model spectrum from Ajello+2008
	# in photons per cm^2 per sec per str per keV
	# Energy in keV
	def dnde_ajello(E, C=10.15e-2, G_1=1.32, G_2=2.88, E_B=29.99):
	    return C/((E/E_B)**G_1+(E/E_B)**G_2)

	# numerical quadrature on the empirical model
	# TODO: Why is this a +-2*sigma square??
	# TODO: integrate using the true instrumental response??
	background, _ = integrate.quad(dnde_ajello, energy*(1.-2*sigma_energy), energy*(1.+2*sigma_energy))
	return background

# Generates a random, uniformly sampled 3D rotation matrix
def generateRotationMatrix():
    
    # First sample the Euler angles using algorithm 1 of Kuffner 2004
    theta = 2*np.pi*np.random.rand()-np.pi # -pi to pi
    eta = 2*np.pi*np.random.rand()-np.pi # -pi to pi
    phi = np.arccos(1.-2*np.random.rand())+0.5*np.pi # 0 to 2*pi
    if np.random.rand() < 0.5:
        if phi < np.pi:
            phi += np.pi
        else:
            phi -= np.pi
    
    # Next, generate the rotation matrix from the Euler angles
    mats = []
    cosz = np.cos(theta)
    sinz = np.sin(theta)
    cosy = np.cos(phi)
    siny = np.sin(phi)
    cosx = np.cos(eta)
    sinx = np.sin(eta)
    mats.append(np.array([[cosz, -sinz, 0],[sinz, cosz, 0],[0, 0, 1]]))
    mats.append(np.array([[cosy, 0, siny],[0, 1, 0],[-siny, 0, cosy]]))
    mats.append(np.array([[1, 0, 0],[0, cosx, -sinx],[0, sinx, cosx]]))
    return reduce(np.dot, mats[::-1])

# nfw density and mass profile
def nfw_rho(r, r_s, rho_0): 
	return rho_0 * (r/r_s)**-1 * (1 + r/r_s)**-2
def nfw_menc(r, r_s, rho_0): 
	return 4*np.pi*rho_0*r_s**3*(np.log((r_s+r)/r_s)-r/(r_s+r))

### Constants: The position and velocity of the Sun, speed of light ###
r_sun = 8.0  # in kpc
pos_sun = r_sun*np.array([1.0, 0.0, 0.0])
vel_sun = np.array([0.0, 220.0, 0.0]) # in km/s
