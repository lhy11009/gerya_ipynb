import numpy as np
import random


class NoMarkerNearPointError(Exception):
    '''
    Error handler class for no markers near a point
    '''
    pass

class MARKER():
    '''
    Material model for marker in cell method
    Attributes:
        xs (list of float): x coordinates of mesh
        ys (list of float): y coordinates of mesh
        n_markers (int): number of markers
        xms (list of float): x coordinates of markers
        yms (list of float): y coordinates of markers
        last_time_increment (float): time increment of last step
    '''
    def __init__(self, mesh, n_markers_x, n_markers_y):
        '''
        Initialization
        Inputs:
            xs (list of float): x coordinates
            ys (list of float): y coordinates
            n_markers_x (int): numbers of markers along x
            n_markers_y (int): numbers of markers along y
            get_indexes (func): take x and y, return an int value of index
        '''
        self.mesh = mesh
        xsize, ysize = mesh.get_sizes()
        xnum, ynum = mesh.get_number_of_points_xy()
        nnode = mesh.get_number_of_points()
        self.n_markers = n_markers_x * n_markers_y
        # marker coordinates
        xms = np.zeros(self.n_markers)
        yms = np.zeros(self.n_markers)
        marker_grid_space_x = xsize / n_markers_x  # use n instead of (n-1) to make sure markers are in the domain
        marker_grid_space_y = ysize / n_markers_y
        for i in range(n_markers_x):
            for j in range(n_markers_y):
                marker_grid_x = (i + 0.5) * marker_grid_space_x
                marker_grid_y = (j + 0.5) * marker_grid_space_y
                i_marker = j + i * n_markers_y
                x_marker = marker_grid_x + (random.uniform(0.0, 1.0) - 0.5) * marker_grid_space_x
                y_marker = marker_grid_y + (random.uniform(0.0, 1.0) - 0.5) * marker_grid_space_y
                xms[i_marker] = x_marker
                yms[i_marker] = y_marker
        self.xms = xms
        self.yms = yms
        self.check_marker_coordinates()
        # marker index
        self.ids = self.get_material_indexes(self.xms, self.yms)
        self.viscosities = [1e21, 1e20]  # relate to index 0 and index 1
        self.densities = [3300.0, 3200.0]
        # field
        self.fields = []
        self.n_field = 2
        self.i_density = 0
        self.i_viscosity = 1
        self.field_averaing = ['arithmetic', 'geometric']
        for i_field in range(self.n_field):
            self.fields.append(np.zeros([ynum, xnum]))

    def set_velocity_method(self, func_vx, func_vy):
        '''
        link to the interface for computing vx and vy
        '''
        self.func_vx = func_vx
        self.func_vy = func_vy
    
    def get_initial_density(self, x, y):
        '''
        Return value of density with coordinates
        '''
        half_width = 50e3
        xsize, ysize = self.mesh.get_sizes()
        if type(x) in [int, float, np.float64] and type(y) in [int, float, np.float64]:
            if (((x - xsize/2.0)**2.0 + (y - ysize/2.0)**2.0)**0.5 < half_width):
                rho = 3300.0
            else:
                rho = 3300.0
        elif type(x) == np.ndarray and type(y) == np.ndarray:
            assert(x.shape == y.shape)
            rho = np.ones(x.shape) * 3300.0
            mask = ((x - xsize/2.0)**2.0 + (y - ysize/2.0)**2.0)**0.5 < half_width
            rho[mask] = 3300.0
        else:
            raise TypeError("Type of x and y should be float or np.ndarray")
        return rho

    def get_initial_viscosity(self, x, y):
        '''
        Return value of viscosity with coordinates, for variable viscosities.
        '''
        half_width = 50e3
        eta_plume = 1e20
        eta_0 = 1e21
        xsize, ysize = self.mesh.get_sizes()
        if type(x) in [int, float, np.float64] and type(y) in [int, float, np.float64]:
            if (((x - xsize/2.0)**2.0 + (y - ysize/2.0)**2.0)**0.5 < half_width):
                eta = eta_plume
            else:
                eta = eta_0
        elif type(x) == np.ndarray and type(y) == np.ndarray:
            assert(x.shape == y.shape)
            eta = np.ones(x.shape) * eta_0
            mask = ((x - xsize/2.0)**2.0 + (y - ysize/2.0)**2.0)**0.5 < half_width
            eta[mask] = eta_plume
        else:
            raise TypeError("Type of x and y should be float or np.ndarray")
        return eta
    
    def get_material_indexes(self, x, y):
        '''
        return material index from coordinates
        '''
        xsize, ysize = self.mesh.get_sizes()
        half_width = 50e3
        if type(x) in [int, float, np.float64] and type(y) in [int, float, np.float64]:
            if (((x - xsize/2.0)**2.0 + (y - ysize/2.0)**2.0)**0.5 < half_width):
                id = 1
            else:
                id = 0
        elif type(x) == np.ndarray and type(y) == np.ndarray:
            assert(x.size == y.size)
            id = [0 for i in range(x.size)]
            id = np.array(id)
            mask = ((x - xsize/2.0)**2.0 + (y - ysize/2.0)**2.0)**0.5 < half_width
            id[mask] = 1
        else:
            raise TypeError("Type of x and y should be float or np.ndarray")
        return id

    
    def assign_velocities_to_markers(self, func_vx, func_vy, **kwargs):
        '''
        Assign velocities to markers
        Inputs:
            func_vx (a function of x and y): a function to get x velocity at x and y
            func_vy (a function of x and y): a function to get y velocity at x and y
            kwargs (dict): 
                accuracy - order of Runge-Kutta approximation to use
                dt - time increment, used for higher order implementation
        '''
        vxs = []
        vys = []
        accuracy = kwargs.get("accuracy", 1)
        dt = kwargs.get("dt", None)
        assert(accuracy in [1, 4])  # first order or fourth order
        for i in range(self.n_markers):
            x = self.xms[i]
            y = self.yms[i]
            if accuracy == 1:
                vx = func_vx(x, y)
                vy = func_vy(x, y)
            elif accuracy == 4:
                vxA = func_vx(x, y)
                vyA = func_vy(x, y)
                xB = x + vxA * dt / 2.0
                yB = y + vyA * dt / 2.0
                vxB = func_vx(xB, yB)
                vyB = func_vy(xB, yB)
                xC = x + vxB * dt / 2.0
                yC = y + vyB * dt / 2.0
                vxC = func_vx(xC, yC)
                vyC = func_vy(xC, yC)
                xD = x + vxC * dt
                yD = y + vyC * dt
                vxD = func_vx(xD, yD)
                vyD = func_vy(xD, yD)
                vx = 1.0 / 6.0 * (vxA + 2.0*vxB + 2.0*vxC + vxD)
                vy = 1.0 / 6.0 * (vyA + 2.0*vyB + 2.0*vyC + vxD)
            else:
                raise ValueError("assign_velocities_to_markers: %d th order accuracy approach is not implemented yet" % accuracy)
            vxs.append(vx)
            vys.append(vy)
        return vxs, vys

    def get_density(self, iy, jx):
        '''
        return viscosity at a point iy, jx
        '''
        density = self.fields[0][iy, jx]
        return density
    
    def export_densities(self):
        '''
        export density field
        '''
        return self.fields[self.i_density]
    
    def get_viscosity(self, iy, jx):
        '''
        return viscosity at a point iy, jx
        '''
        eta = self.fields[1][iy, jx]
        return eta

    def export_viscosities(self):
        '''
        export viscosity field
        '''
        return self.fields[self.i_viscosity]

    def get_viscosity_at_P(self, iy, jx):
        '''
        return the viscosity defined at the P node of a cell
        '''
        xs, ys = self.mesh.get_coordinates()
        eta = self.get_viscosity(iy, jx)
        etaUL = self.get_viscosity(iy-1, jx-1)
        etaA = self.get_viscosity(iy, jx-1)
        etaL = self.get_viscosity(iy-1, jx)
        eta1 = average(eta, etaUL, etaA, etaL, method='harmonic')
        return eta1
    
    def advect(self, accuracy, last_time_increment):
        '''
        advect the marker field
        Return:
            dt (float): time step
            accuracy (int): accuracy of the Runge-Kutta method
            last_time_increment(float): time increment of the last step
        '''
        xsize, ysize = self.mesh.get_sizes()
        vxs, vys = self.assign_velocities_to_markers(self.func_vx, self.func_vy, accuracy=accuracy, dt=last_time_increment)
        # get the time step
        dt = 0.0
        is_first = True
        for i in range(self.n_markers):
            x = self.xms[i]
            y = self.yms[i]
            x_stp, y_stp = self.mesh.get_cell_size(x, y)
            vx =  vxs[i]
            vy = vys[i]
            dt_i = min(abs(x_stp/vx/2.0), abs(y_stp/vy/2.0))
            if is_first:
                dt = dt_i
                is_first = False
            else:
                dt = min(dt, dt_i)
        # advect
        print("Advecting by dt = %.4e" % dt)
        for i in range(self.n_markers):
            x0 = self.xms[i] 
            x1 = x0 + vxs[i] * dt 
            y0 = self.yms[i] 
            y1 = y0 + vys[i] * dt
            if x1 > xsize:
                x1 -= xsize
            elif x1 < 0.0:
                x1 += xsize
            if y1 > ysize:
                y1 -= ysize
            elif y1 < 0.0:
                y1 += ysize
            self.xms[i] = x1
            self.yms[i] = y1
        return dt
            
    def get_markers(self):
        '''
        Return x and y coordinates of markers
        '''
        return self.xms, self.yms, self.ids
    
    def check_marker_coordinates(self):
        '''
        check every marker is within domain
        '''
        xsize, ysize = self.mesh.get_sizes()
        assert(np.min(self.xms) > 0.0 and np.max(self.xms) < xsize)
        assert(np.min(self.yms) > 0.0 and np.max(self.yms) < ysize)

    
    def execute(self):
        '''
        export densities on mesh nodes
        '''
        nnode = self.mesh.get_number_of_points()
        xnum, ynum = self.mesh.get_number_of_points_xy()
        xs, ys = self.mesh.get_coordinates()
        weights = np.zeros([ynum, xnum])
        counts = np.zeros([ynum, xnum])
        # make fields 0.0
        for i in range(self.n_field):
            self.fields[i] = np.zeros([ynum, xnum])
        # compute weights and add up values
        for i_marker in range(self.n_markers):
            xm = self.xms[i_marker]
            ym = self.yms[i_marker]
            node0, _, _, _ = self.mesh.node_index(xm, ym)
            iy, jx = self.mesh.get_indexes_of_node(node0)
            x0 = xs[jx]
            x1 = xs[jx+1]
            y0 = ys[iy]
            y1 = ys[iy+1]
            weight_x0 = (xm - x1) / (x0 - x1)
            weight_x1 = (xm - x0) / (x1 - x0)
            weight_y0 = (ym - y1) / (y0 - y1)
            weight_y1 = (ym - y0) / (y1 - y0)
            weight00 = weight_y0 * weight_x0
            weight01 = weight_y0 * weight_x1
            weight11 = weight_y1 * weight_x1
            weight10 = weight_y1 * weight_x0
            counts[iy, jx] += 1
            counts[iy, jx+1] += 1
            counts[iy+1, jx+1] +=1
            counts[iy+1, jx] += 1
            weights[iy, jx] += weight00
            weights[iy, jx+1] += weight01
            weights[iy+1, jx+1] += weight11
            weights[iy+1, jx] += weight10
            self.fields[0][iy, jx] += weight00 * self.densities[self.ids[i_marker]]
            self.fields[1][iy, jx] += weight00 / self.viscosities[self.ids[i_marker]]
            self.fields[0][iy, jx+1] += weight01 * self.densities[self.ids[i_marker]]
            self.fields[1][iy, jx+1] += weight01 / self.viscosities[self.ids[i_marker]]
            self.fields[0][iy+1, jx+1] += weight11 * self.densities[self.ids[i_marker]]
            self.fields[1][iy+1, jx+1] += weight11 / self.viscosities[self.ids[i_marker]]
            self.fields[0][iy+1, jx] += weight10 * self.densities[self.ids[i_marker]]
            self.fields[1][iy+1, jx] += weight10 / self.viscosities[self.ids[i_marker]]
        # assert every node has at least one marker
        for iy in range(ynum):
            for jx in range(xnum):
                if counts[iy, jx] < 1.0:
                    raise NoMarkerNearPointError("MARKER.execute: point %d (iy) %d (jx) has no adjacent markers" % (iy, jx))
        # average fields
        for i_field in range(self.n_field):
            if self.field_averaing[i_field] == 'arithmetic':
                self.fields[i_field] = self.fields[i_field] / weights
            elif self.field_averaing[i_field] == 'geometric':
                self.fields[i_field] = weights / self.fields[i_field]


def average(*vars, **kwargs):
    '''
    average vars by the method given
    '''
    method = kwargs.get('method', 'arithmatic')
    aver = 0
    if method == "arithmatic":
        n = 0
        for var in vars:
            aver += var
            n += 1
        aver = aver/n
    elif method == "harmonic":
        n = 0
        for var in vars:
            assert(np.abs(var) > 1e-16)
            aver += 1.0 / var
            n += 1
        aver = n / aver
    else:
        raise TypeError("Method must be arithmatic or harmonic")
    return aver