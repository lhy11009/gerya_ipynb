from python_scripts.GMesh2d import *
import numpy as np

def test_init():
	'''
	Test initiate a mesh and get the coordinates
	Assert:
		the get_coordinates() method should return the same point coordinates
	'''
	xsize = 500000.0 # Model size, m
	ysize = 750000.0
	xnum = 51   # Number of nodes
	ynum = 76
	xstp = xsize/(xnum-1) # Grid step
	ystp = ysize/(ynum-1)
	xs = np.linspace(0.0, xsize, xnum) # construct xs 
	ys = np.linspace(0.0, ysize, ynum) # construct ys
	Mesh2d = MESH2D(xs, ys)
	mesh_xs, mesh_ys = Mesh2d.get_coordinates()
	assert(mesh_xs.size == 51)
	assert(mesh_ys.size == 71)