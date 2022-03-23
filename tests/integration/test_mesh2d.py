import numpy as np
from matplotlib import pyplot as plt
from python_scripts.GMesh2d import *
from matplotlib import pyplot as plt
import os

test_dir = ".test"
if not os.path.isdir(test_dir):
	os.mkdir(test_dir)


def test_get_vx_vy_nodes():
	'''
	Give the coordinates of a point, plot the 4 vx nodes and the 4 vy nodes that contains this point.
	test this with a smaller mesh
	'''
	xsize = 500000.0 # Model size, m
	ysize = 750000.0
	xnum = 5   # Number of nodes
	ynum = 7
	xstp = xsize/(xnum-1) # Grid step
	ystp = ysize/(ynum-1)
	xs = np.linspace(0.0, xsize, xnum) # construct xs 
	ys = np.linspace(0.0, ysize, ynum) # construct ys
	Mesh2d = MESH2D(xs, ys)

	mesh_xs, mesh_ys = Mesh2d.get_coordinates() # coordinates
	xx, yy = np.meshgrid(mesh_xs, mesh_ys)
	fig = plt.figure(tight_layout=True, figsize=(5, 5))
	# gs = gridspec.GridSpec(3, 1)

	# Give the coordinates of a point, plot the 4 vx nodes and the 4 vy nodes that contains this point.
	query_x = 499.0e3 # figure 0: check vx and vy points
	query_y = 699.0e3
	vx_xs, vx_ys, _ = Mesh2d.get_vx_cell_nodes(query_x, query_y)
	vy_xs, vy_ys, _ = Mesh2d.get_vy_cell_nodes(query_x, query_y)
	print(vy_xs, vy_ys)
	# plot test result
	ax = fig.add_subplot()  
	h=ax.scatter(xx/1e3, yy/1e3, color='black')
	h1=ax.scatter(query_x/1e3, query_y/1e3, color='red')
	h2=ax.scatter(vx_xs/1e3, vx_ys/1e3, color='green')
	h3=ax.scatter(vy_xs/1e3, vy_ys/1e3, color='blue')
	ax.set_xlim([-xstp/1e3, (xstp + xsize)/1e3])
	ax.set_ylim([-ystp/1e3, (ystp + ysize)/1e3])
	ax.set_xlabel('X (km)') 
	ax.set_ylabel('Y (km)')
	ax.invert_yaxis()
	save_path = os.path.join(test_dir, "get_vx_vy_nodes.png")
	if os.path.isfile(save_path):
		os.remove(save_path)  # remove old files
	plt.savefig(save_path)
	assert(os.path.isfile(save_path))