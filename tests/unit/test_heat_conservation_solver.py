import numpy as np
from python_scripts.GHeatConservationSolver import *
from python_scripts.GMesh2d import MESH2D


def rectangular_wave_temperature(x, y, xsize, ysize):
		'''
		temperature profile of a retangular wave in the middle
		'''
		dx = xsize / 10.0
		dy = ysize / 10.0
		if type(x) == float and type(y) == float:
			assert(x >= 0 and x <= xsize and y >= 0 and y <= ysize)
			if x > (xsize - dx) / 2.0 and x < (xsize + dx) / 2.0\
				and y > (ysize - dy) / 2.0 and y < (ysize + dy) / 2.0:
				T = 1300.0
			else:
				T = 1000.0
		elif type(x) == np.ndarray and type(y) == np.ndarray:
			assert(x.shape == y.shape)
			mask = (x > (xsize - dx) / 2.0) & (x < (xsize + dx) / 2.0)\
				& (y > (ysize - dy) / 2.0) & (y < (ysize + dy) / 2.0)
			T = np.ones(x.shape) * 1000.0
			T[mask] = 1300.0
		else:
			raise TypeError("Type of x or y is wrong (either float or numpy.ndarray")
		return T


def test_explicit_constant_k():
	'''
	test a problem of constant thermal expansivity
	'''
	# initiate mesh
	xsize = 1000000.0 # Model size, m
	ysize = 1500000.0
	xnum = 6   # Number of nodes
	ynum = 4
	xs = np.linspace(0.0, xsize, xnum) # construct xs 
	ys = np.linspace(0.0, ysize, ynum) # construct ys
	Mesh2d = MESH2D(xs, ys)
	# initial temperature
	xxs, yys = np.meshgrid(xs, ys)
	Ts_init = rectangular_wave_temperature(xxs, yys, xsize, ysize)
	print(Ts_init.shape)  # debug
	# initiate solver
	HCSolver = EXPLICIT_SOLVER(Mesh2d, use_constant_thermal_conductivity=True)
	HCSolver.assemble(3.0, 3200.0, 1000.0, 1.0)
	HCSolver.solve()

