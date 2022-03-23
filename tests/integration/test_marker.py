# test the implementation for MARKER

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from python_scripts.GMarker import *
from python_scripts.GMesh2d import MESH2D  # mesh class to use
from matplotlib import pyplot as plt
import os

test_dir = ".test"
if not os.path.isdir(test_dir):
	os.mkdir(test_dir)


def test_marker_composition():
	# test a smaller mesh
	xsize = 500000.0 # Model size, m
	ysize = 750000.0
	xnum = 5   # Number of nodes
	ynum = 7
	xstp = xsize/(xnum-1) # Grid step
	ystp = ysize/(ynum-1)
	xs = np.linspace(0.0, xsize, xnum) # construct xs 
	ys = np.linspace(0.0, ysize, ynum) # construct ys
	Mesh2d = MESH2D(xs, ys)

	Marker = MARKER(Mesh2d, 20, 30)
	xms, yms, ids = Marker.get_markers()
	fig = plt.figure(tight_layout=True, figsize=(5, 10))
	gs = gridspec.GridSpec(2, 1)
	ax = fig.add_subplot(gs[0, 0])  # test 1: plot the position of all markers
	h=ax.scatter(xms/1e3, yms/1e3)
	ax.set_xlim([0.0, xsize/1e3])
	ax.set_ylim([0.0, ysize/1e3])
	ax.set_xlabel('X (km)') 
	ax.set_ylabel('Y (km)')
	ax.invert_yaxis()
	ax = fig.add_subplot(gs[1, 0])  # test 2: color the markers by their composition (blue for composition 1, red for composition 2)
	# (composition 2 is in the middle of the domain)
	colors = ['c', 'r']
	h=ax.scatter(xms/1e3, yms/1e3, c=[colors[id] for id in ids])
	ax.set_xlim([0.0, xsize/1e3])
	ax.set_ylim([0.0, ysize/1e3])
	ax.set_xlabel('X (km)') 
	ax.set_ylabel('Y (km)')
	ax.invert_yaxis()
	save_path = os.path.join(test_dir, "get_vx_vy_nodes.png")
	if os.path.isfile(save_path):
		os.remove(save_path)  # remove old files
	plt.savefig(save_path)
	assert(os.path.isfile(save_path))