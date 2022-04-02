import numpy as np
import time
import scipy.sparse as sparse
import scipy.sparse.linalg



class HC_SOLVER():
    def __init__(self):
        pass
    
    def initial_temperature(self, Ts):
        '''
        set initial temperature
        Inputs:
            Ts (ndarray) - array of temperature
        '''
        assert(Ts.shape==(self.Ny, self.Nx))
        # We want Ts to first increment on y axis and then on x axis,
        # thus we want to first transpose Ts, so that y would be the
        # first to go when we rival the matrix
        self.Ts = Ts.T.reshape(self.Nx * self.Ny)
    
    def get_time(self):
        '''
        get time outputs
        return
            time (float)
        '''
        return self.t

    def solve(self):
        '''
        solve the linear equations
        '''
        assert(self.assembled==True)
        start = time.time()
        self.Ts = scipy.sparse.linalg.spsolve(self.L, self.R)
        self.solved = True  # reset the flags
        self.assembled = False
        self.t += self.dt
        end = time.time()
        time_elapse = end - start
        print("Temperature solver: %.4e s to solver" % time_elapse)
    
    def global_index(self, iy, jx):
        '''
        Get global index from the indexing along x and y, note the differences to Geyra's book
        as we start from index 0
        Inputs:
            iy (int) - y index
            jx (int) - x index
        '''
        return self.Ny * jx + iy

    def export(self):
        '''
        export results, in the form of meshed data
        Return:
            xxs (ndarray of float): x coordinates
            yys (ndarray of float): y coordinates
            Ts (ndarray of float): temperature
        '''
        xs, ys = self.mesh.get_coordinates()
        xxs, yys = np.meshgrid(xs, ys)
        # here, again, we want the increment to first follow the y axis.
        Ts = np.transpose(self.Ts.reshape((self.Nx, self.Ny)))
        return xxs, yys, Ts


class IMPLICIT_SOLVER(HC_SOLVER):

    def __init__(self, mesh, **kwargs):
        '''
        Initiation.
        Inputs:
            mesh (MESH2D object)
            kwargs (dict):
                'use_constant_thermal_conductivity' - use constant thermal conductivity in the model
                    by default, this is False. This would affect the choices of solver used.
        '''
        self.mesh = mesh
        Nx, Ny = mesh.get_number_of_points_xy()
        self.Nx = Nx
        self.Ny = Ny
        self.Ts = np.zeros((Nx * Ny))
        self.assembled = False
        self.solved = False
        self.t = 0.0
        pass


    def thermal_profile(self, rhos, cps, thermal_conductivities):
        '''
        set initial temperature
        Inputs:
            Ts (ndarray) - array of temperature
        '''
        assert(rhos.shape==(self.Ny, self.Nx))
        assert(cps.shape==(self.Ny, self.Nx))
        assert(thermal_conductivities.shape==(self.Ny, self.Nx))
        # We want Ts to first increment on y axis and then on x axis,
        # thus we want to first transpose Ts, so that y would be the
        # first to go when we rival the matrix
        self.rhos = rhos.T.reshape(self.Nx * self.Ny)
        self.cps = cps.T.reshape(self.Nx * self.Ny)
        self.thermal_conductivities = thermal_conductivities.T.reshape(self.Nx * self.Ny)
      

    def assemble(self, dt, **kwargs):
        '''
        assemble the equations
        Inputs:
            thermal_conductivity - thermal conductivities
            rhos - densities
            cps - heat capacity
            dt (float): increment in time
            kwargs (dict)
        '''
        # guess is, grid size is 1 km. This won't affect the stability of the solution
        scaling1 = kwargs.get('scaling', 1e-6 / (1000.0)**2.0 * 1000.0)
        assert(self.assembled == False)
        self.dt = dt
        Ts = self.Ts.copy()
        xs, ys = self.mesh.get_coordinates()
        I = []  # These are indexed into the "left" matrix L
        J = []  # of the linear function Lx = R
        V = []  # equivalent to L(i, j) = v
        R = []
        i = 0  # incrememnt on i for I indexes
        for jx in range(self.Nx):
            for iy in range(self.Ny):
                k1 = self.global_index(iy, jx - 1)   # index of the point to the left
                k2 = self.global_index(iy - 1, jx)   # index of the point above
                k3 = self.global_index(iy, jx)  # index of this point
                k4 = self.global_index(iy + 1, jx)   # index of the point below
                k5 = self.global_index(iy, jx + 1)   # index of the point to the right
                rho = self.rhos[k3]
                cp = self.cps[k3]
                if iy > 0 and iy < self.Ny-1 and jx > 0 and jx < self.Nx-1:
                    # internal points
                    # derive kappa at these points
                    tc1 = self.thermal_conductivities[k1]
                    tc2 = self.thermal_conductivities[k2]
                    tc3 = self.thermal_conductivities[k3]
                    tc4 = self.thermal_conductivities[k4]
                    tc5 = self.thermal_conductivities[k5]
                    # increment in x and y
                    dx = xs[jx + 1] - xs[jx]
                    dy = ys[iy + 1] - ys[iy]
                    I.append(i)  # manage entry of T1
                    J.append(k1)
                    v1 = - (tc1 + tc3) / (2 * dx**2.0) / scaling1
                    V.append(v1)
                    I.append(i)  # manage entry of T2
                    J.append(k2)
                    v2 = - (tc2 + tc3) / (2 * dy**2.0) / scaling1
                    V.append(v2)
                    I.append(i)  # manage entry of T3
                    J.append(k3)
                    v3 = (rho * cp / dt + (tc3 + tc5) / (2 * dx**2.0)\
                        + (tc1 + tc3) / (2 * dx**2.0) + (tc3 + tc4) / (2 * dy**2.0)\
                        + (tc2 + tc3) / (2 * dy**2.0)) / scaling1
                    V.append(v3)
                    I.append(i)  # manage entry of T4
                    J.append(k4)
                    v4 = - (tc3 + tc4) / (2 * dy**2.0) / scaling1
                    V.append(v4)
                    I.append(i)  # manage entry of T5
                    J.append(k5)
                    v5 = - (tc3 + tc5) / (2 * dx**2.0) / scaling1
                    V.append(v5)
                    r = rho * cp * Ts[k3] / dt / scaling1
                    R.append(r)  # entry on the right
                elif iy == 0:
                    # Hereby, the boundaries are handled here. The insulating boudnary conditions
                    # are used here. This is formulated by one point on the boundary and another interal point
                    # next to it.
                    # top boundary
                    I.append(i)
                    J.append(k3)
                    V.append(1)
                    I.append(i)
                    J.append(k4)
                    V.append(-1)
                    R.append(0.0)
                elif iy == self.Ny - 1:
                    # bottom boundary
                    I.append(i)
                    J.append(k2)
                    V.append(1)
                    I.append(i)
                    J.append(k3)
                    V.append(-1)
                    R.append(0.0)
                elif jx == 0:
                    # left boundary
                    I.append(i)
                    J.append(k3)
                    V.append(1)
                    I.append(i)
                    J.append(k5)
                    V.append(-1)
                    R.append(0.0)
                elif jx == self.Nx - 1:
                    # right boundary
                    I.append(i)
                    J.append(k1)
                    V.append(1)
                    I.append(i)
                    J.append(k3)
                    V.append(-1)
                    R.append(0.0)
                else:
                    raise IndexError("Index error found with iy = %d, jx = %d" % (iy, jx))
                i += 1
        assert(i == self.Nx * self.Ny)
        # Finally, assemble the L matrix and the R vector.
        self.L = sparse.csr_matrix((V, (I, J)), shape=(self.Nx * self.Ny, self.Nx * self.Ny))
        self.R = np.array(R)
        self.assembled = True  # set the flags
        self.solved = False
    

class EXPLICIT_SOLVER(HC_SOLVER):

    def __init__(self, mesh, **kwargs):
        '''
        Initiation.
        Inputs:
            mesh (MESH2D object)
            kwargs (dict):
                'use_constant_thermal_conductivity' - use constant thermal conductivity in the model
                    by default, this is False. This would affect the choices of solver used.
        '''
        self.use_constant_thermal_conductivity = kwargs.get('use_constant_thermal_conductivity', False)
        self.mesh = mesh
        Nx, Ny = mesh.get_number_of_points_xy()
        self.Nx = Nx
        self.Ny = Ny
        self.Ts = np.zeros((Nx * Ny))
        self.assembled = False
        self.solved = False
        self.t = 0.0
        pass
    
    def assemble(self, thermal_conductivity, rho, cp, dt):
        '''
        assemble the equations
        Inputs:
            thermal_conductivity - thermal conductivities
            rho - densities
            cp - heat capacity
            dt (float): increment in time
        '''
        assert(self.assembled == False)
        self.dt = dt
        Ts = self.Ts.copy()
        xs, ys = self.mesh.get_coordinates()
        I = []  # These are indexed into the "left" matrix L
        J = []  # of the linear function Lx = R
        V = []  # equivalent to L(i, j) = v
        R = []
        if self.use_constant_thermal_conductivity:
            ### constant k, rho and cp
            kappa = thermal_conductivity / (rho * cp)
            i = 0  # incrememnt on i for I indexes
            for jx in range(self.Nx):
                for iy in range(self.Ny):
                    k1 = self.global_index(iy, jx - 1)   # index of the point to the left
                    k2 = self.global_index(iy - 1, jx)   # index of the point above
                    k3 = self.global_index(iy, jx)  # index of this point
                    k4 = self.global_index(iy + 1, jx)   # index of the point below
                    k5 = self.global_index(iy, jx + 1)   # index of the point to the right
                    if iy > 0 and iy < self.Ny-1 and jx > 0 and jx < self.Nx-1:
                        # internal points
                        dx = xs[jx + 1] - xs[jx]
                        dy = ys[iy + 1] - ys[iy]
                        Ts_new = Ts[k3] + kappa * dt * ( (Ts[k1] - 2 * Ts[k3] + Ts[k5]) / dx**2.0\
                             + (Ts[k2] - 2*Ts[k3] + Ts[k4]) / dy**2.0) # get the solution directly
                        I.append(i)
                        J.append(k3)
                        V.append(1)
                        R.append(Ts_new)
                    elif iy == 0:
                        # Hereby, the boundaries are handled here. The insulating boudnary conditions
                        # are used here. This is formulated by one point on the boundary and another interal point
                        # next to it.
                        # top boundary
                        I.append(i)
                        J.append(k3)
                        V.append(1)
                        I.append(i)
                        J.append(k4)
                        V.append(-1)
                        R.append(0.0)
                    elif iy == self.Ny - 1:
                        # bottom boundary
                        I.append(i)
                        J.append(k2)
                        V.append(1)
                        I.append(i)
                        J.append(k3)
                        V.append(-1)
                        R.append(0.0)
                    elif jx == 0:
                        # left boundary
                        I.append(i)
                        J.append(k3)
                        V.append(1)
                        I.append(i)
                        J.append(k5)
                        V.append(-1)
                        R.append(0.0)
                    elif jx == self.Nx - 1:
                        # right boundary
                        I.append(i)
                        J.append(k1)
                        V.append(1)
                        I.append(i)
                        J.append(k3)
                        V.append(-1)
                        R.append(0.0)
                    else:
                        raise IndexError("Index error found with iy = %d, jx = %d" % (iy, jx))
                    i += 1
            assert(i == self.Nx * self.Ny)
            # Finally, assemble the L matrix and the R vector.
            self.L = sparse.csr_matrix((V, (I, J)), shape=(self.Nx * self.Ny, self.Nx * self.Ny))
            self.R = np.array(R)
            self.assembled = True  # set the flags
            self.solved = False