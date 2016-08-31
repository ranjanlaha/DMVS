import numpy as np
import time
import healpy as hp
import scipy.integrate as integrate
import ctypes
from ctypes import *


# load and initialize the c library
_hps = ctypes.CDLL('speedyVS/speedyvs.so')

# struct for passing constant data
class vsParams(Structure):

	_fields_ = [("rsun", c_double), ("vsun", c_double), ("Rs", c_double), ("rho0", c_double), 
			("b", c_double), ("l", c_double), ("theta", c_double), 
			("rsun2", c_double), ("cosb", c_double), ("cosl", c_double), 
			("rmax", c_double), ("omega", c_double), ]
	
	def __init__(self, rsun=8.0, vsun=220.0, Rs=19.78, rho0=6.912562e6, \
			b=25.0, l=25.0, theta=20.0):

		# geometry of the halo and NFW params 
		self.rsun = rsun # kpc 
		self.vsun = vsun # km/s
		self.Rs = Rs # kpc

		# assumed to be in M_sun/kpc^3
		# convert to keV/cm^3
		self.rho0 = rho0*3.7966e-5 
		#print self.rho0
	
		# LOS and FOV params
		torad = np.pi/180.0
		self.b = b*torad
		self.l = l*torad
		self.theta = theta*torad # degrees

		# compute secondary geometric stuff
		self.rsun2 = self.rsun**2
		self.cosb = np.cos(self.b)
		self.cosl = np.cos(self.l)
		self.rmax = 10000.0 #200.0 # TODO: check if this is large enough
		self.omega = 0.0

	def from_param(self):
		return byref(self) 

# arg and return types
_hps.rho_nfw_los.argtypes = [c_double, POINTER(vsParams)]
_hps.rho_nfw_los.restype = c_double
_hps.adaptive_quadrature.argtypes = [POINTER(vsParams)]
_hps.adaptive_quadrature.restype = c_double

_hps.aq_fov.argtypes = [POINTER(vsParams)]
_hps.aq_fov.restype = c_double

_hps.py_integral.argtypes = (c_int, c_double)
_hps.py_integral.restype = (c_double)


if __name__ == '__main__':

	print "Testing c functions..."
	print "Integrating the J-factor numerically..."

	vsp = vsParams()
	def rfo(s):
		return _hps.rho_nfw_los(s, vsp)

	print "\n= scipy.integrate line-integral ="
	tstart = time.time()
	jfac, err = integrate.quad(rfo, 0.0, vsp.rmax)
	tend = time.time()
	print ' J-factor = ', jfac, "err = ", err
	print ' dt = %.5e' % (tend-tstart)

	print "\n= C adaptive quadrature line-integral ="
	tstart = time.time()
	jfac = _hps.adaptive_quadrature(vsp)
	tend = time.time()
	print ' J-factor = ', jfac
	print ' dt = %.5e' % (tend-tstart)

	print "\n= scipy.integrate 3D FOV ="
	tstart = time.time()
	_hps.py_set_default_params()	
	jfac, err = integrate.nquad(_hps.py_integral, ranges=((0.0, vsp.rmax), (0.0, vsp.theta), (0.0, 2*np.pi)))
	#jfac, err = integrate.nquad(_hps.py_integral, ranges=((0.0, 1.0), (0.0, 1.0), (0.0, 2*np.pi)))
	tend = time.time()
	omega = 2*np.pi*(1.0-np.cos(vsp.theta));
	jfac /= omega #* vsp.rmax 
	err /= omega #* vsp.rmax
	print ' J-factor = ', jfac, "err = ", err
	print ' dt = %.5e' % (tend-tstart)


	print "\n= C adaptive quadrature 3D FOV ="
	tstart = time.time()
	jfac = _hps.aq_fov(vsp)
	tend = time.time()
	print ' J-factor = ', jfac
	print ' dt = %.5e' % (tend-tstart)

	print "\n"




