import numpy as np
import time
#import healpy as hp
import scipy.integrate as integrate
import ctypes
from ctypes import *

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

# struct for passing constant data
npoint_sigma = 2048
class vsParams(Structure):

	_fields_ = [("rsun", c_double), ("vsun", c_double), 
			("Rs", c_double), ("rho0", c_double), ("Rvir", c_double), 
			("b", c_double), ("l", c_double), ("theta", c_double), ("rmax", c_double), 
			("cosb", c_double), ("cosl", c_double), ("sinb", c_double), ("sinl", c_double),
			("sigv", c_double * (npoint_sigma+1)), ("dr", c_double),]
	
	def __init__(self, rsun=8.0, vsun=220.0, Rs=19.78, Rvir=276.3, rho0=6.912562e6, \
			b=25.0, l=25.0, theta=20.0):

		# geometry of the halo and NFW params 
		self.rsun = rsun # kpc 
 		# TODO: this is a hacky inversion!!!
		self.vsun = -vsun # km/s
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

# main test routine
if __name__ == '__main__':

	# setup parameters 
	vsp = vsParams(theta=20.0, l=65.0)
	quadargs = {'ranges': ((0.0, vsp.rmax), (0.0, vsp.theta), (0.0, 2*np.pi)),}

	print "\n= Integrating the J-factor numerically using scipy.integrate.nquad ="

	tstart = time.time()
	jfac, err = integrate.nquad(svslib.py_nfw_polarproj_rho, **quadargs)
	tend = time.time()
	print ' j-factor = %f, err = %.5e, ncall = %d, dt = %.5e' % (jfac, err, 0, tend-tstart)

	tstart = time.time()
	vavg, err = integrate.nquad(svslib.py_nfw_polarproj_vel, **quadargs)
	tend = time.time()
	vavg /= jfac
	err /= jfac
	print ' mean velocity = %f, err = %.5e, ncall = %d, dt = %.5e' % (vavg, err, 0, tend-tstart)

	tstart = time.time()
	sigv, err = integrate.nquad(svslib.py_nfw_polarproj_sigv, args=[vavg], **quadargs)
	tend = time.time()
	sigv /= jfac
	err /= jfac
	sigv **= 0.5
	print ' line width = %f, err = %.5e, ncall = %d, dt = %.5e' % (sigv, err, 0, tend-tstart)

	print "\n"




'''
####### deprecated code ########

# wrapper for C LOS implementation
svslib.aq_los.argtypes = (POINTER(c_double), POINTER(c_double), \
		POINTER(c_int), c_double, POINTER(vsParams),)
svslib.aq_los.restype = (c_int);
def integrateLOS(vsp, tol=1.0e-4):
	result = c_double() 
	err = c_double()
	ncall = c_int()
	if not svslib.aq_los(byref(result), byref(err), byref(ncall), tol, vsp):
		raise RuntimeError("Stack overflow in integration\n") 
	return result.value, err.value, ncall.value

# wrapper for C FOV implementation
svslib.aq_fov.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_int), \
		c_double, POINTER(vsParams))
svslib.aq_fov.restype = (c_int);
def integrateFOV(vsp, tol=1.0e-4):
	result = c_double() 
	err = c_double()
	ncall = c_int()
	if not svslib.aq_fov(byref(result), byref(err), byref(ncall), tol, vsp):
		raise RuntimeError("Stack overflow in integration\n") 
	return result.value, err.value, ncall.value




	#print "\n= C adaptive quadrature line-integral ="
	#tstart = time.time()
	#jfac, err, ncall = integrateLOS(vsp)
	#tend = time.time()
	#jfac *= omega
	#err*= omega
	#print ' j-factor = %f, err = %.5e, ncall = %d' % (jfac, err, ncall)
	#print ' dt = %.5e' % (tend-tstart)

	#print "\n= C adaptive quadrature 3D FOV ="
	#tstart = time.time()
	#jfac, err, ncall = integrateFOV(vsp, tol=1.0e2)
	#tend = time.time()
	#print ' j-factor = %f, err = %.5e, ncall = %d' % (jfac, err, ncall)
	#print ' dt = %.5e' % (tend-tstart)



	#print "\n= scipy.integrate line-integral ="
	#tstart = time.time()
	#jfac, err = integrate.quad(svslib.py_nfw_los, 0.0, vsp.rmax)
	#tend = time.time()
	#jfac *= omega
	#err*= omega
	#print ' j-factor = %f, err = %.5e, ncall = %d, dt = %.5e' % (jfac, err, 0, tend-tstart)
	#print info



'''


