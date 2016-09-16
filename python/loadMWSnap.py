#!/lustre/ki/pfs/dmpowel1/anaconda/bin/python
import matplotlib.pyplot as plt
import numpy as np
from readGadgetSnapshot import readGadgetSnapshot

#allHaloes = [268, 288, 374, 414, 416, 460, 490, 558, 570, 628, 749, 878, 881, 990, 530, 800,\
		#852, 926, 937, 9829, 415, 364, 327, 825, 829, 939, 8247, 23, 247, 797, 925, 119, 567, 675,\
		#14, 469, 641, 569, 573, 9749, 349, 606, 88, 188, 270, 738, 718, 967, 440]

# all the haloes that retain the proper output data files
allHaloes = [268, 288, 374, 414, 416, 460, 490, 558, 570, 628, 878, 881, 530, 800,\
		852, 926, 937, 9829, 415, 8247, 567, 675, 469, 641, 9749, 349, 967]
baseDir = '/nfs/slac/g/ki/ki21/cosmo/yymao/mw_resims/halos'

def loadMWSnap(halo=937, snap=235, verbose=True):

	# check whether the halo snapnum is valid
	if halo not in allHaloes:
		raise ValueError("Halo %d does not exist!"%halo)
	if snap > 235:
		raise ValueError("Snapshot %d does not exist!"%snap)

	# load Yao's MW resims
	snapname = '%s/Halo%03d/output/snapshot_%03d' % (baseDir, halo, snap)
	#rockname = '%s/Halo%03d/rockstar/out_%03d.list' % (baseDir, halo, snap) 
	musicname = '%s/Halo%03d/music.conf_log.txt' % (baseDir, halo)
	targethaloname = '%s/Halo%03d/analysis/target_halo.txt' % (baseDir, halo)
	hlistname = '%s/Halo%03d/rockstar/hlists/hlist_1.00000.list' % (baseDir, halo)


	# load the target halo data from the first line of hlists
	targid = np.loadtxt(targethaloname, usecols=(0,), dtype=np.int64)
	if verbose:
		print "Reading rockstar output", hlistname 
	with open(hlistname) as f:
		for line in f:
			if line.startswith('#'):
				continue
			'''
			#scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6) desc_pid(7) phantom(8)
			sam_mvir(9) mvir(10) rvir(11) rs(12) vrms(13) mmp?(14) scale_o
			f_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21) vz(22) Jx(23) Jy(24) Jz(25) Spin(26)
			Breadth_first_ID(27) Depth_first_ID(28) Tree_root_ID(29) Orig
			_halo_ID(30) Snap_num(31) Next_coprogenitor_depthfirst_ID(32) Last_progenitor_depthfirst_ID(33)
			Rs_Klypin(34) Mvir_all(35) M200b(36) M200c(37) M500c(38) 
			M2500c(39) Xoff(40) Voff(41) Spin_Bullock(42) b_to_a(43) c_to_a(44) A[x](45) A[y](46) A[z](47)
			b_to_a(500c)(48) c_to_a(500c)(49) A[x](500c)(50) A[y](500c
			)(51) A[z](500c)(52) T/|U|(53) Macc(54) Mpeak(55) Vacc(56) Vpeak(57) Halfmass_Scale(58)
			Acc_Rate_Inst(59) Acc_Rate_100Myr(60) Acc_Rate_Tdyn(61)
			'''
			halodata = np.genfromtxt([line], usecols=(1,10,11,34,15,17,18,19,20,21,22,43,44), 
				dtype=np.dtype([('id', np.int64), ('mvir', np.float64), ('rvir', np.float64), ('rs', np.float64),
					('last_mm_scale', np.float64), ('cm', np.float64, (3,)), ('cv', np.float64,
						(3,)), ('b_to_a', np.float64), ('c_to_a', np.float64)]))
			
			# stop if we have found the correct halo
			if halodata['id'] == targid:
				break

	# check that we have found the correct target halo
	if targid != halodata['id']:
		raise ValueError("Target halo ID not found!")
	cm = halodata['cm']
	cv = halodata['cv']

	# read MUSIC output to identify the refined patch dimensions
	if verbose:
		print "Reading MUSIC log", musicname 
	with open(musicname) as f:
		for l in f:
			if 'Level  13 :   offset' in l:
				l = f.next()
				ldim = tuple(map(lambda s: int(s.strip().rstrip(')')), l.partition('(')[2].split(',')))
				break
	if verbose:
		print "Lagrangian patch dimensions are", ldim

	# read the snapshot
	if verbose:
		print "Reading Gadget snapshot", snapname 
	part_type = 1 # dark matter particles only
	
	for i in xrange(8):
		header, pos, vel, ids = readGadgetSnapshot('%s.%d'%(snapname, i),
				read_pos=True, read_vel=True, read_id=True, single_type=part_type)
		if i == 0:
			hubble = header.HubbleParam
			mpp = header.mass[part_type]
			nptot = header.npartTotal[part_type]
			assert nptot == np.product(ldim)
			allpos = np.empty((nptot, 3), dtype=np.float64)
			allvel = np.empty((nptot, 3), dtype=np.float64)

		# subtract off halo CM and CV so that returned values are 
		# relative to the halo center
		allpos[ids] = pos-cm
		allvel[ids] = vel-cv
	
	# put in physical units
	halodata['mvir'] /= hubble
	halodata['rvir'] /= hubble
	halodata['rs'] /= hubble
	#c = halodata['rvir']/halodata['rs']
	#halodata['rho0'] = halodata['mvir']/(4*np.pi*halodata['rs']**3*(np.log(1+c)-c/(1+c))) 
	allpos *= 1.0e3/hubble
	mpp *= 1.0e10/hubble

	# return in physical units
	# position in physical kpc, velocity in km/s, particle mass in physical m_sun
	return allpos, allvel, mpp, ldim, halodata


