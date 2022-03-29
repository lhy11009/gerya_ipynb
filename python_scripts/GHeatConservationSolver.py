import numpy as np
import time
import scipy.sparse as sparse
import scipy.sparse.linalg


class IMPLICIT_SOLVER():
    def __init__(self):
        pass


class EXPLICIT_SOLVER():

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