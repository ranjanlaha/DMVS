#!/lustre/ki/pfs/dmpowel1/anaconda/bin/python -u

import numpy as np
import healpy as hp
import time
import scipy.integrate as integrate
import ctypes
from ctypes import *

import loadMWSnap as mws

# load and initialize the c library
svslib = ctypes.CDLL('speedyVS/speedyvs.so')
svslib.py_nfw_polarproj_rho.argtypes = (c_int, c_double)
svslib.py_nfw_polarproj_rho.restype = (c_double)
svslib.py_nfw_polarproj_vel.argtypes = (c_int, c_double)
svslib.py_nfw_polarproj_vel.restype = (c_double)
svslib.py_nfw_polarproj_sigv.argtypes = (c_int, c_double)
svslib.py_nfw_polarproj_sigv.restype = (c_double)
svslib.py_sigv2_integrand.argtypes = (c_int, c_double)
svslib.py_sigv2_integrand.restype = (c_double)
svslib.rho_nfw.argtypes = (c_double,)
svslib.rho_nfw.restype = (c_double)
svslib.py_nbody.argtypes = (POINTER(c_double), POINTER(c_double), c_int,
		POINTER(c_double), POINTER(c_double), POINTER(c_double))

svslib.py_nbody_photons.argtypes = (POINTER(c_double), POINTER(c_double), c_int,
		c_double, POINTER(c_double), POINTER(c_int),)

# struct for passing constant data to the c lib
npoint_sigma = 2048
class vsParams(Structure):

	_fields_ = [("pos_sun", c_double * 3), ("vel_sun", c_double * 3), 
			("Rs", c_double), ("rho0", c_double), ("Rvir", c_double), 
			("b", c_double), ("l", c_double), ("theta", c_double), ("rmax", c_double), 
			("cosb", c_double), ("cosl", c_double), ("sinb", c_double), ("sinl", c_double),
			("sigv", c_double * (npoint_sigma+1)), ("dr", c_double),]
	
	def __init__(self, pos_sun=(8.0, 0.0, 0.0), vel_sun=(0.0, 220.0, 0.0), \
			Rs=19.78, Rvir=276.3, rho0=6.912562e6, \
			b=25.0, l=25.0, theta=20.0):

		# geometry of the halo and NFW params 
		self.pos_sun[0] = pos_sun[0]; # kpc
		self.pos_sun[1] = pos_sun[1];
		self.pos_sun[2] = pos_sun[2];
		self.vel_sun[0] = vel_sun[0]; # km/s
		self.vel_sun[1] = vel_sun[1];
		self.vel_sun[2] = vel_sun[2];
		self.Rs = Rs # kpc
		self.rho0 = rho0 # M_sun/kpc^3
		self.Rvir = Rvir # kpc
	
		# LOS and FOV params
		torad = np.pi/180.0
		self.b = b*torad
		self.l = l*torad
		self.theta = theta*torad # degrees
		self.rmax = 10000.0 

		# compute secondary geometric stuff
		self.cosb = np.cos(self.b)
		self.cosl = np.cos(self.l)
		self.sinb = np.sin(self.b)
		self.sinl = np.sin(self.l)

		svslib.py_set_params(self)	

		# compute the velocity dispersion
		G = 4.302e-6 # in the correct units of kpc M_sun^-1 (km/s)^2
		r = np.linspace(0.0, self.Rvir, npoint_sigma+1)
		self.sigv[0] = 0.0 # avoid the singularity at r=0
		for i in xrange(1, npoint_sigma):
			self.sigv[i], err = integrate.quad(svslib.py_sigv2_integrand, r[i], self.Rvir)
			self.sigv[i] /= svslib.rho_nfw(r[i]) 
			self.sigv[i] *= G
			self.sigv[i] **= 0.5
		self.sigv[npoint_sigma] = 0.0 
		self.dr = self.Rvir/npoint_sigma
		svslib.py_set_params(self)	

	def setLOS(self, b=25.0, l=25.0):
	
		torad = np.pi/180.0
		self.b = b*torad
		self.l = l*torad
		self.cosb = np.cos(self.b)
		self.cosl = np.cos(self.l)
		self.sinb = np.sin(self.b)
		self.sinl = np.sin(self.l)
		svslib.py_set_params(self)	

	def from_param(self):
		return byref(self) 

svslib.py_set_params.argtypes = (POINTER(vsParams),)


# compute the observed DM decay line centroid uncertainty for the given N-body data and
# instrumental response parameters
def spectroscopy(mode='analytic', Rs=19.78, Rvir=276.3, rho0=6.912562e6,
		pos=None, vel=None, mass=None, lon=[-85., -65., -45., -25., 0., 25., 45., 65., 85.], lat=25.0,
		aperture=1.0, exposure=300.0, sigma_energy=0.0003639950, theta_fov=20.0):
	
	### Sterile neutrino parameters from Bulbul+2014 ###
	# TODO: Assumes 3.5 keV observing energy for now!!!
	en = 3.55 # keV photons
	m_x = 7.1 # mass in keV
	Gamma = (1.38e-29)*(7e-4)*(m_x)**5 # decay rate in s^-1

	# Get the number of background photons for this observation 
	fov = 2*np.pi*(1.0-np.cos(theta_fov*np.pi/180.0)) # fov in str
	background = aperture*exposure*fov*integrate_background(en, sigma_energy)

	#print 'bg at 3.5*(1-2*10^-3) kev =', dnde_ajello(3.55*(1.-2*10**-3))
	#print 'bg at 3.5*(1+2*10^-3) kev =', dnde_ajello(3.55*(1.+2*10**-3))
	
	# mash all model parameters into a prefactor with correct unit conversions, 
	# to get the photon number count into the detector
	prefac = Gamma*aperture*exposure/(4.*np.pi*m_x) # decay rate, aperture, exposure time, solid angle
	prefac *= 1.12e63 # number of sterile neutrinos per simulation particle
	prefac *= 1.050265e-43 # put aperture in units of kpc^2

	# setup parameters
	vsp = vsParams(pos_sun=pos_sun, vel_sun=vel_sun, Rs=Rs, Rvir=Rvir, rho0=rho0, theta=theta_fov)

	# sample and plot
	lonr = np.array(lon)
	cenr = np.zeros_like(lonr)
	siglr = np.zeros_like(lonr)
	siger = np.zeros_like(lonr)
	sigcr = np.zeros_like(lonr)
	crr = np.zeros_like(lonr)
	nsr = np.zeros_like(lonr)
	nbr = np.zeros_like(lonr)
	for i, l in enumerate(lon): 

		vsp.setLOS(b=lat, l=l)

		if mode == 'nbody':
			# sample the flux, mean velocity, and dispersion from particle data
			jfac = c_double(0.0)
			center = c_double(0.0)
			sigma_line = c_double(0.0)
			svslib.py_nbody(pos.ctypes.data_as(POINTER(c_double)),
				vel.ctypes.data_as(POINTER(c_double)), pos.shape[0], 
				byref(jfac), byref(center), byref(sigma_line))
			jfac = jfac.value*mass
			center = center.value
			sigma_line  = sigma_line.value

		if mode == 'analytic':
			# numerically integrate the flux, mean velocity, and dispersion
			# using an NFW halo and velocity dispersion model
			quadargs = {'ranges': ((0.0, vsp.rmax), (0.0, vsp.theta), (0.0, 2*np.pi)),}
			jfac, err = integrate.nquad(svslib.py_nfw_polarproj_rho, **quadargs)
			center, err = integrate.nquad(svslib.py_nfw_polarproj_vel, **quadargs)
			center /= jfac
			sigma_line, err = integrate.nquad(svslib.py_nfw_polarproj_sigv, args=[center], **quadargs)
			sigma_line /= jfac
			sigma_line **= 0.5

		# get the photon number count for the given exposure
		# put in units of c
		Ns = prefac*jfac
		center /= 3.0e5
		sigma_line /= 3.0e5
		
		# effective line width when convolved with Astro-H's energy resolution
		# Cramer-Rao bound for backgrounds
		# the error in the centroid according to Poisson stats
		sigma_eff = (sigma_line**2 + sigma_energy**2)**0.5
		C_R = cramer_rao(sig=Ns, bg=background)
		sigma_center = sigma_eff*C_R*Ns**-0.5

		# append to return values
		cenr[i] = center
		siglr[i] = sigma_line
		siger[i] = sigma_eff
		sigcr[i] = sigma_center
		crr[i] = C_R 
		nsr[i] = Ns 
		nbr[i] = background 

		# print a little progress indicator
		print '.', 

	print ''
	ret_dict = {'l': lonr, 'centroid': cenr, 'sigma_line': siglr, 'sigma_eff': siger,
			'sigma_centroid': sigcr, 'cr': crr, 'ns': nsr, 'nb': nbr}
	return ret_dict 

def samplePhotonsFromData(data, do_bg=True):

	cent = data['centroid']
	sigma = data['sigma_eff'] # TODO: how to incorporate sigma_instr properly? Last??
	meanphot = data['ns']
	if do_bg:
		# add background counts and spread according to Cramer-Rao
		sigma *= data['cr']
		meanphot += data['nb']

	# sample photons and get line statistics 
	nphot = np.random.poisson(lam=meanphot)
	nsamp = np.sum(nphot)
	photons = np.empty((2, np.sum(nphot)))
	line = np.empty((3, len(data['l'])))
	nsamp = 0
	for i, l in enumerate(data['l']):

		# sample photons for this l
		en = np.random.normal(loc=cent[i], scale=sigma[i], size=nphot[i])
		photons[0, nsamp:nsamp+nphot[i]] = l 
		photons[1, nsamp:nsamp+nphot[i]] = en
		nsamp += nphot[i]

		# get line statistics for this l
		line[0, i] = l # pointing
		line[1, i] = np.mean(en) # centroid
		line[2, i] = np.std(en)*nphot[i]**-0.5 # std. deviation of the centroid 

	return photons, line 

# compute the observed DM decay line centroid uncertainty for the given N-body data and
# instrumental response parameters
def samplePhotonsFromNBody(pos, vel, mass, 
		lon=[-85., -65., -45., -25., 0., 25., 45., 65., 85.], lat=25.0,
		aperture=1.0, exposure=300.0, sigma_energy=0.0003639950, theta_fov=20.0):

	
	### Sterile neutrino parameters from Bulbul+2014 ###
	# TODO: Assumes 3.5 keV observing energy for now!!!
	en = 3.55 # keV photons
	m_x = 7.1 # mass in keV
	Gamma = (1.38e-29)*(7e-4)*(m_x)**5 # decay rate in s^-1

	# TODO: how to treat backgrounds??
	# Get the number of background photons for this observation 
	#fov = 2*np.pi*(1.0-np.cos(theta_fov*np.pi/180.0)) # fov in str
	#background = aperture*exposure*fov*integrate_background(en, sigma_energy)
	
	# mash all model parameters into a prefactor with correct unit conversions, 
	# to get the photon number count into the detector
	prefac = Gamma*aperture*exposure/(4.*np.pi*m_x) # decay rate, aperture, exposure time, solid angle
	prefac *= 1.12e63 # put solar masses in units of keV 
	prefac *= 1.050265e-43 # put aperture in units of kpc^2
	prefac *= mass

	# setup parameters
	vsp = vsParams(pos_sun=pos_sun, vel_sun=vel_sun, theta=theta_fov)


	# sample and plot
	fudge = 1 #6
	prefac *= fudge
	photons = np.empty((2, 512*fudge))
	pbuf = np.empty(128*fudge)
	line = np.empty((3, len(lon)))
	nsamp = 0
	for i, l in enumerate(lon): 

		# sample photons
		vsp.setLOS(b=lat, l=l)
		nbuf = c_int(0)
		svslib.py_nbody_photons(pos.ctypes.data_as(POINTER(c_double)),
			vel.ctypes.data_as(POINTER(c_double)), pos.shape[0], prefac,
			pbuf.ctypes.data_as(POINTER(c_double)), byref(nbuf)) 
		nbuf = nbuf.value
		pbuf[:nbuf] /= 3.0e5 # put in units of c 
		photons[0, nsamp:nsamp+nbuf] = l 
		photons[1, nsamp:nsamp+nbuf] = pbuf[:nbuf] 
		nsamp += nbuf

		# get line statistics for this l
		line[0, i] = l # pointing
		line[1, i] = np.mean(pbuf[:nbuf]) # centroid
		sigma_eff = ((np.std(pbuf[:nbuf])**2+sigma_energy**2)/nbuf)**0.5 # uncertainty in the centroid 
		line[2, i] = sigma_eff 
		#print '.', 

	#print ''
	print '.', 

	return photons[:,:nsamp], line

		   

# Optimal Cramer-Rao bound on uncertainty with backgrounds
def cramer_rao(sig=1, bg=0):
    return (1.+4.*(bg/sig))**0.5

# CXB model spectrum from Ajello+2008
# in photons per cm^2 per sec per str per keV
# Energy in keV
def dnde_ajello(E, C=10.15e-2, G_1=1.32, G_2=2.88, E_B=29.99):
    return C/((E/E_B)**G_1+(E/E_B)**G_2)

def integrate_background(energy, sigma_energy):

	# numerical quadrature on the empirical model
	# TODO: Why is this a +-2*sigma square??
	# TODO: integrate using the true instrumental response??
	background, _ = integrate.quad(dnde_ajello, energy*(1.-2*sigma_energy), energy*(1.+2*sigma_energy))
	return background

# nfw density and mass profile
def nfw_rho(r, r_s, rho_0): 
	return rho_0 * (r/r_s)**-1 * (1 + r/r_s)**-2
def nfw_menc(r, r_s, rho_0): 
	return 4*np.pi*rho_0*r_s**3*(np.log((r_s+r)/r_s)-r/(r_s+r))

### Constants: The position and velocity of the Sun, speed of light ###
r_sun = 8.0  # in kpc
pos_sun = r_sun*np.array([1.0, 0.0, 0.0])
vel_sun = np.array([0.0, 220.0, 0.0]) # in km/s


# i/o routines
def writeVSData(halo, mode, data):
	fname = '../data/halo%d.%s' % (halo, mode)
	ndata = np.array([len(data['l'])], dtype=np.int32)
	with open(fname, 'wb') as f:
		ndata.tofile(f)
		data['l'].tofile(f)
		data['centroid'].tofile(f)
		data['sigma_line'].tofile(f)
		data['sigma_eff'].tofile(f)
		data['sigma_centroid'].tofile(f)
		data['cr'].tofile(f)
		data['ns'].tofile(f)
		data['nb'].tofile(f)
		ndata.tofile(f)
		f.close()

def readVSData(halo, mode):
	fname = '../data/halo%d.%s' % (halo, mode)
	data = {}
	with open(fname, 'rb') as f:
		ndata = np.fromfile(f, dtype=np.int32, count=1)[0]
		data['l'] = np.fromfile(f, dtype=np.float64, count=ndata)
		data['centroid'] = np.fromfile(f, dtype=np.float64, count=ndata)
		data['sigma_line'] = np.fromfile(f, dtype=np.float64, count=ndata)
		data['sigma_eff'] = np.fromfile(f, dtype=np.float64, count=ndata)
		data['sigma_centroid'] = np.fromfile(f, dtype=np.float64, count=ndata)
		data['cr'] = np.fromfile(f, dtype=np.float64, count=ndata)
		data['ns'] = np.fromfile(f, dtype=np.float64, count=ndata)
		data['nb'] = np.fromfile(f, dtype=np.float64, count=ndata)
		assert ndata == np.fromfile(f, dtype=np.int32, count=1)[0]
		f.close()
	return data

# runs velocity spectroscopy on all available haloes!
if __name__ == '__main__':

	### Micro-X parameters ###
	fwhm_MX = 3.0/(3.5*10**3) # fractional energy resolution at 3.5 keV
	mx_params = {
		'aperture': 1.0, # effective aperture in cm^2
		'exposure': 300.0, # exposure time in sec
		'sigma_energy': fwhm_MX/(2.*(2.*np.log(2.))**0.5), # Guassian sigma of the energy response
		'theta_fov': 20.0, # angular radius of the sampling cone in degrees
	}
	print 'Micro-X params =', mx_params 
	mx_lon = np.array([-105., -85., -65., -45., -25., 0., 25., 45., 65., 85., 105.])
	fill_lon = np.linspace(-120.0, 120.0, 50)
	
	# process all haloes using analytic vs n-body
	haloes = mws.allHaloes
	for i, h in enumerate(haloes):

		# load the halo data 
		print 'Halo %d of %d (%d):' % (i+1, len(haloes), h)
		print ' Loading N-body data...' 
		pos, vel, partmass, ldim, data = mws.loadMWSnap(halo=h, verbose=False)

		# solve for rho0 for NFW profile
		c = data['rvir']/data['rs']
		rho0=data['mvir']/(4*np.pi*data['rs']**3*(np.log(1+c)-c/(1+c))) 

		# sample N-body, analytic, and analytic for fill plotting
		print ' N-body sampling',
		nbody = spectroscopy(mode='nbody', 
				pos=pos, vel=vel, mass=partmass, lon=mx_lon, **mx_params)
		print ' Analytic sampling (N-body pointings)',
		analytic = spectroscopy(mode='analytic', 
				rho0=rho0, Rs=data['rs'], Rvir=data['rvir'], lon=mx_lon, **mx_params)
		print ' Analytic sampling (fill plot)',
		fill = spectroscopy(mode='analytic', 
				rho0=rho0, Rs=data['rs'], Rvir=data['rvir'], lon=fill_lon, **mx_params)

		# write to file
		writeVSData(h, 'analytic', analytic)
		writeVSData(h, 'nbody', nbody)
		writeVSData(h, 'fill', fill)

'''

# runs velocity spectroscopy on all available haloes!
if __name__ == '__main__':

	### Micro-X parameters ###
	fwhm_MX = 3.0/(3.5*10**3) # fractional energy resolution at 3.5 keV
	mx_params = {
		'aperture': 1.0, # effective aperture in cm^2
		'exposure': 300.0, # exposure time in sec
		'sigma_energy': fwhm_MX/(2.*(2.*np.log(2.))**0.5), # Guassian sigma of the energy response
		'theta_fov': 20.0, # angular radius of the sampling cone in degrees
	}
	print 'Micro-X params =', mx_params 
	mx_lon = np.array([-105., -85., -65., -45., -25., 0., 25., 45., 65., 85., 105.])
	
	# process all haloes using analytic vs n-body
	haloes = [374]
	#haloes = haloesWithVSData 
	for i, h in enumerate(haloes):

		# load the halo data 
		print 'Halo %d of %d (%d):' % (i+1, len(haloes), h)
		print ' Loading N-body data...' 
		pos, vel, partmass, ldim, data = mws.loadMWSnap(halo=h, verbose=False)

		# sample N-body, analytic, and analytic for fill plotting
		print ' N-body sampling',
		nbody = spectroscopy(mode='nbody', pos=pos, vel=vel, 
				mass=partmass, lon=mx_lon, **mx_params)
		print nbody

		print ' Analytic sampling',
		c = data['rvir']/data['rs']
		rho0=data['mvir']/(4*np.pi*data['rs']**3*(np.log(1+c)-c/(1+c))) 
		analytic = spectroscopy(mode='analytic', 
				rho0=rho0, Rs=data['rs'], Rvir=data['rvir'], lon=mx_lon, **mx_params)
		print analytic

		print ' N-body sampling of photons',
		print samplePhotonsFromNBody(pos, vel, partmass, lon=mx_lon, **mx_params)
		#print nbody


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


# filters particle data into a cone based on latitude and longitude of the los
# along with the angular radius of the sampling cone
# default los is the fiducial observing direction from Speckhard+2016
# returns the LOS velocities and their distances
def sample_los(pos, vel, lon=20.0, lat=25.0, beam_radius=20.0):
    
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
 




'''
