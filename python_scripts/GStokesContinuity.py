import numpy as np

class STOKES_CONTINUITY_MARKERS():
    '''
    '''  
    def __init__(self, mesh, MaterialModel, **kwargs):
        '''
        initialization
        Attributes:
            xs (list of float): x coordinates
            ys (list of float): y coordinates
            MaterialModel(class object): interface for materials
            xnum (int): number of points along x
            ynum (int): number of points along y
            N: number of degree of freedoms
            L: left matrix to solve
            R: right matrix to solve
            assembled (bool): matrix is assembled
            solved (bool): solution is derived
            dofs (dict): degree of freedoms
            t (float): time
            step (int): step
        kwargs (dict):
            use_marker: use marker for the material model
        '''
        xs = 0
        ys = 0 # debug
        self.mesh = mesh
        self.MaterialModel = MaterialModel
        self.is_material_model_marker = kwargs.get('use_marker', False)
        # print(self.mesh.get_coordinates())
        xs, ys = self.mesh.get_coordinates()
        self.xnum = xs.size
        self.ynum = ys.size
        self.N = 3 * (self.xnum + 1) * (self.ynum + 1)  # number of the unknown, 3 (vx, vy, P) * number of node.
        self.L = None
        self.R = None
        self.assembled = False
        self.solved = False
        self.S = None
        self.dofs = {}
        self.t = 0.0 # time
        self.step = 0
        # last time step
        self.last_time_increment = None

    def assemble(self):
        '''
        Assemble the matrix
        '''
        start = time.time()
        self.MaterialModel.execute()
        I = []
        J = []
        V = []
        R = np.zeros(self.N) # the right matrix
        k = 0
        # nondimensionalization
        eta0 = self.MaterialModel.get_viscosity_at_P(1, 1)  # estimate of the viscosity
        xs, ys = self.mesh.get_coordinates()
        xstp = xs[1] - xs[0]  # estimate of resolution along x, only consider uniform grid for now
        ystp = ys[1] - ys[0]
        Kcont = 2 * eta0 / (xstp + ystp)   
        self.Kcont = Kcont  # record this for parse the P solution 
        scale = eta0 / xstp ** 2.0 # use this to scale the entry
        scale1 = Kcont / xstp
        # fill in internal cells
        # loop for points, first along y axis, then along x axis.
        # Incement on k to fill in entries in each row in the L matrix,
        # as well as the R vector
        nStokes_x = 0
        nStokes_y = 0
        nContinuity = 0
        nBdVx = 0
        nBdVy = 0
        nGhost = 0
        for jx in range(xnum+1):
            for iy in range(ynum+1):
                # value of x and y
                x = jx * xstp
                y = iy * ystp
                inode0 =self.get_node_index(iy, jx)
                inodeA =self.get_node_index(iy-1, jx)
                inodeB =self.get_node_index(iy+1, jx)
                inodeL =self.get_node_index(iy, jx-1)
                inodeR =self.get_node_index(iy, jx+1)
                inodeUR =self.get_node_index(iy-1, jx+1)
                inodeBL =self.get_node_index(iy+1, jx-1)
                if iy > 0 and iy < ynum and jx > 0 and jx < xnum-1:      
                    # x-Stokes equation
                    ivx1, _, _ = self.get_indexed_on_node(inodeL)
                    ivx2, ivy1, _ = self.get_indexed_on_node(inodeA)
                    ivx3, ivy2, ip1 = self.get_indexed_on_node(inode0)
                    ivx4, _, _ = self.get_indexed_on_node(inodeB)
                    ivx5, ivy4, ip2 = self.get_indexed_on_node(inodeR)
                    _, ivy3, _ = self.get_indexed_on_node(inodeUR)
                    eta = self.MaterialModel.get_viscosity_at_P(iy, jx)  # get viscosity
                    etaA = self.MaterialModel.get_viscosity(iy-1, jx)
                    etaR = self.MaterialModel.get_viscosity_at_P(iy, jx+1)
                    etaB = self.MaterialModel.get_viscosity(iy, jx)
                    I.append(k)  # vx node to the left
                    J.append(ivx1)
                    V.append(2.0 * eta / xstp ** 2.0 / scale)
                    I.append(k)  # vx node above
                    J.append(ivx2)
                    V.append(etaA / ystp ** 2.0 / scale)
                    I.append(k)  # vx node
                    J.append(ivx3)
                    V.append((- 2.0 * etaR / xstp**2.0 - 2.0 * eta / xstp**2.0\
                        - etaB / ystp**2.0 - etaA / ystp**2.0)/ scale)
                    I.append(k)  # vx node below
                    J.append(ivx4)
                    V.append(etaB / ystp ** 2.0 / scale)
                    I.append(k) # vx node to the right
                    J.append(ivx5)
                    # V.append(eta / xstp ** 2.0 / scale)
                    V.append(2.0 * etaR / xstp ** 2.0 / scale)
                    I.append(k) # vy node above
                    J.append(ivy1)
                    V.append(etaA / xstp / ystp / scale)
                    I.append(k) # vy node
                    J.append(ivy2)
                    V.append(-etaB / xstp / ystp / scale)
                    I.append(k) # vy node to the upper right
                    J.append(ivy3)
                    V.append(-etaA / xstp / ystp / scale)
                    I.append(k) # vy node to the right
                    J.append(ivy4)
                    V.append(etaB / xstp/ ystp/ scale)
                    I.append(k)  # P node
                    J.append(ip1)
                    V.append(Kcont / xstp / scale)
                    I.append(k) # P node to the right
                    J.append(ip2)
                    V.append(-Kcont / xstp / scale)
                    R[k] = 0
                    k += 1
                    nStokes_x += 1
                if iy > 0 and iy < ynum-1 and jx > 0 and jx < xnum:
                    # y-Stokes equation
                    ivx1, ivy1, _ = self.get_indexed_on_node(inodeL)
                    _, ivy2, _ = self.get_indexed_on_node(inodeA)
                    ivx3, ivy3, ip1 = self.get_indexed_on_node(inode0)
                    ivx4, ivy4, ip2 = self.get_indexed_on_node(inodeB)
                    _, ivy5, _ = self.get_indexed_on_node(inodeR)
                    ivx2, _, _ = self.get_indexed_on_node(inodeBL)
                    eta = self.MaterialModel.get_viscosity_at_P(iy, jx)  # get viscosity
                    etaB = self.MaterialModel.get_viscosity_at_P(iy+1, jx)
                    etaL = self.MaterialModel.get_viscosity(iy, jx-1)
                    etaR = self.MaterialModel.get_viscosity(iy, jx)
                    I.append(k) # vy node to the left
                    J.append(ivy1)
                    V.append(etaL / xstp ** 2.0 / scale)
                    I.append(k)  # vy node above
                    J.append(ivy2)
                    V.append(2.0 * eta / ystp ** 2.0 / scale)
                    I.append(k) # vy node
                    J.append(ivy3)
                    V.append((- 2.0 * etaB / ystp**2.0 - 2.0 * eta / ystp**2.0\
                        -etaR / xstp**2.0 - etaL / xstp**2.0) / scale)
                    I.append(k)  # vy node below
                    J.append(ivy4)
                    V.append(2.0 * etaB / ystp ** 2.0 / scale)
                    I.append(k)  # vy node to the right
                    J.append(ivy5)
                    V.append(etaR / xstp ** 2.0 / scale)
                    I.append(k) # vx node to the left
                    J.append(ivx1)
                    V.append(etaL / xstp / ystp / scale)
                    I.append(k) # vx node to the bottom-left
                    J.append(ivx2)
                    V.append(-etaL / xstp / ystp / scale)
                    I.append(k) # vx node
                    J.append(ivx3)
                    V.append(-etaR / xstp / ystp / scale)
                    I.append(k) # vx node below
                    J.append(ivx4)
                    V.append(etaR / xstp / ystp / scale)
                    I.append(k) # P node
                    J.append(ip1)
                    V.append(Kcont / ystp / scale)
                    I.append(k) # P node below
                    J.append(ip2)
                    V.append(-Kcont / ystp / scale)
                    R[k] = -g / 2.0 * (self.MaterialModel.get_density(iy, jx-1) + self.MaterialModel.get_density(iy,jx)) / scale
                    k += 1
                    nStokes_y += 1
                if iy > 0 and iy < ynum and jx > 0 and jx < xnum and not (iy == 1 and jx == 1):
                    # continuity equation
                    # note: check the last condition with others
                    ivx1, _, _ = self.get_indexed_on_node(inodeL)
                    _, ivy1, _ = self.get_indexed_on_node(inodeA)
                    ivx2, ivy2, _ = self.get_indexed_on_node(inode0)
                    I.append(k)  # vx node to the left
                    J.append(ivx1)
                    V.append(-Kcont / xstp / scale1)
                    I.append(k) # vx node
                    J.append(ivx2)
                    V.append(Kcont / xstp / scale1)
                    I.append(k) # vy node above
                    J.append(ivy1)
                    V.append(-Kcont / ystp / scale1)
                    I.append(k) # vy node
                    J.append(ivy2)
                    V.append(Kcont / ystp / scale1)
                    R[k] = 0 
                    k += 1
                    nContinuity += 1
                if iy not in [0, ynum] and jx in [0, xnum-1]:
                    # boudnary conditions by vx, normal
                    ivx, _, _ = self.get_indexed_on_node(inode0)
                    I.append(k)
                    J.append(ivx)
                    V.append(1.0)
                    R[k] = 0
                    k += 1
                    nBdVx += 1
                if iy in [0, ynum-1] and jx != xnum:
                    # boudnary conditions by vx, tangential
                    ivx1, _, _ = self.get_indexed_on_node(inode0)
                    ivx2, _, _ = self.get_indexed_on_node(inodeB)
                    I.append(k)  # vx node to the left
                    J.append(ivx1)
                    V.append(1.0)
                    I.append(k)  # vx node
                    J.append(ivx2)
                    V.append(-1.0)
                    R[k] = 0
                    k += 1
                    nBdVx += 1
                if iy in [0, ynum-1] and jx not in [0, xnum]:
                    # boundary conditions by vy, normal
                    _, ivy, _ = self.get_indexed_on_node(inode0)
                    I.append(k)
                    J.append(ivy)
                    V.append(1.0)
                    R[k] = 0
                    k += 1
                    nBdVy += 1
                if iy != ynum and jx in [0, xnum-1]:
                    # boundary conditions by vy, normal
                    _, ivy1, _ = self.get_indexed_on_node(inode0)
                    _, ivy2, _ = self.get_indexed_on_node(inodeR)
                    I.append(k)  # vy node
                    J.append(ivy1)
                    V.append(1.0)
                    I.append(k)  # vy node to the right
                    J.append(ivy2)
                    V.append(-1.0)
                    R[k] = 0
                    k += 1
                    nBdVy += 1
                if jx == xnum:
                    # ghost points for vx
                    ivx, _, _ = self.get_indexed_on_node(inode0)
                    I.append(k)
                    J.append(ivx)
                    V.append(1.0)
                    R[k] = 0
                    k += 1
                    nGhost += 1
                if iy == ynum:
                    # ghost points for vy
                    _, ivy, _ = self.get_indexed_on_node(inode0)
                    I.append(k)
                    J.append(ivy)
                    V.append(1.0)
                    R[k] = 0
                    k += 1
                    nGhost += 1
                if jx in [0, xnum] or iy in [0, ynum] or (iy == 1 and jx == 1):
                    # ghost points for P
                    # note:check last condition with people
                    _, _, ip = self.get_indexed_on_node(inode0)
                    I.append(k)
                    J.append(ip)
                    V.append(1.0)
                    R[k] = 0
                    k += 1
                    nGhost += 1 
        self.dofs['x stokes'] = nStokes_x
        self.dofs['y stokes'] = nStokes_y
        self.dofs['continuity'] = nContinuity
        self.dofs['bc vx'] = nBdVx
        self.dofs['bc vy'] = nBdVy
        self.dofs['ghost'] = nGhost
        self.dofs['total'] = k
        print(self.dofs)
        assert(k == self.N) # check we covered each row in the matrix
        self.L = sparse.csr_matrix((V, (I, J)), shape=(self.N, self.N))
        self.R = R
        self.assembled = True
        self.solved = False
        end = time.time()
        time_elapse = end - start
        print("Stokes solver: %.4e s to assemble" % time_elapse)

    def solve(self):
        '''
        solve the linear equations
        '''
        start = time.time()
        self.S = scipy.sparse.linalg.spsolve(self.L, self.R)
        self.solved = True
        end = time.time()
        time_elapse = end - start
        print("Stokes solver: %.4e s to solver" % time_elapse)
    
    def advect(self):
        '''
        composition advection
        '''
        start = time.time()
        assert(self.is_material_model_marker)  # first make sure we use marker for material model
        self.MaterialModel.set_velocity_method(self.get_vx, self.get_vy)
        if self.step == 0:
            dt = self.MaterialModel.advect(1, None)
        else:
            print("time increment for the last step:", self.last_time_increment) # debug
            print(type(self.last_time_increment))
            assert(self.last_time_increment > 0.0)
            dt = self.MaterialModel.advect(4, self.last_time_increment)
        self.t += dt
        self.last_time_increment = dt
        self.step += 1
        end = time.time()
        time_elapse = end - start
        print("Solver: %.4e s to advect" % time_elapse)
    
    def get_time(self):
        '''
        Return:
            t (float): time
            step (int): step
        '''
        return self.t, self.step

    def parse_solution_P(self):
        '''
        parse the solution of P from S
        Returns:
            xx (x coordinates of P points)
            yy (y coordinates of P points)
            PP (pressures)
        '''
        xx = np.zeros((self.xnum-1, self.ynum-1))
        yy = np.zeros((self.xnum-1, self.ynum-1))
        PP = np.zeros((self.xnum-1, self.ynum-1))
        for jx in range(1, self.xnum):
            for iy in range(1, self.ynum):
                # internal node
                inode0 =self.get_node_index(iy, jx)
                _, _, ip = self.get_indexed_on_node(inode0)
                x = (xs[jx] + xs[jx-1]) / 2.0
                y = (ys[iy] + ys[iy-1]) / 2.0
                xx[jx-1, iy-1] = x
                yy[jx-1, iy-1] = y
                PP[jx-1, iy-1] = self.S[ip] * self.Kcont
        return xx, yy, PP

    def parse_solution_Vx(self):
        '''
        parse the solution of vx from S
        Returns:
            xx (x coordinates of Vx points)
            yy (y coordinates of Vx points)
            vvx (x velocity)
        '''
        xx = np.zeros((self.xnum, self.ynum-1))
        yy = np.zeros((self.xnum, self.ynum-1))
        vvx = np.zeros((self.xnum, self.ynum-1))
        for jx in range(0, self.xnum):
            for iy in range(1, self.ynum):
                # internal node
                inode0 =self.get_node_index(iy, jx)
                ivx, _, _ = self.get_indexed_on_node(inode0)
                x = xs[jx]
                y = (ys[iy] + ys[iy-1]) / 2.0
                xx[jx, iy-1] = x
                yy[jx, iy-1] = y
                vvx[jx, iy-1] = self.S[ivx]
        return xx, yy, vvx   

    def parse_solution_Vy(self):
        '''
        parse the solution of vy from S
        Returns:
            xx (x coordinates of Vy points)
            yy (y coordinates of Vy points)
            vvy (y velocity)
        '''
        xx = np.zeros((self.xnum-1, self.ynum))
        yy = np.zeros((self.xnum-1, self.ynum))
        vvy = np.zeros((self.xnum-1, self.ynum))
        for jx in range(1, self.xnum):
            for iy in range(0, self.ynum):
                # internal enode
                inode0 =self.get_node_index(iy, jx)
                _, ivy, _ = self.get_indexed_on_node(inode0)
                x = (xs[jx] + xs[jx-1]) / 2.0
                y = ys[iy]
                xx[jx-1, iy] = x
                yy[jx-1, iy] = y
                vvy[jx-1, iy] = self.S[ivy]
        return xx, yy, vvy 

    def get_node_index(self, iy, jx):
        '''
        indexing for nodes
        Inputs:
            iy (int): y index
            jx (int): x index
        Returns:
            inode (int): index of node
        '''
        inode = jx * (ynum + 1) + iy
        return inode
        
    def get_indexed_on_node(self, inode):
        '''
        indexing for P, vx, vy nodes
        Inputs:
            inode (int): node index
        Returns:
            ivx (index of vx)
            ivy (index of vy)
            ip (index of p)
        '''
        ivx = 3 * inode
        ivy = 3 * inode + 1
        ip = 3 * inode + 2
        return ivx, ivy, ip
    
    def get_vy(self, x, y):
        '''
        Return the y velocity from vy points
        '''
        assert(self.solved)
        assert(self.mesh.is_in_mesh(x, y))  # assert point in domain

        xs0, ys0, inodes0 = self.mesh.get_vy_cell_nodes(x, y)
        x0 = xs0[0]
        x1 = xs0[1]
        y0 = ys0[0]
        y1 = ys0[2]
        vs0 = []
        for inode in inodes0:
            _, ivy, _ = self.get_indexed_on_node(inode)
            vs0.append(self.S[ivy])
        vy = BilinearInterpolation(x, y, x0, x1, y0, y1, vs0)
        return vy

    def get_vx(self, x, y):
        '''
        Return the x velocity from vx points
        '''
        assert(self.solved)
        assert(self.mesh.is_in_mesh(x, y))  # assert point in domain

        xs0, ys0, inodes0 = self.mesh.get_vx_cell_nodes(x, y)
        x0 = xs0[0]
        x1 = xs0[1]
        y0 = ys0[0]
        y1 = ys0[2]
        vs0 = []
        for inode in inodes0:
            ivx, _, _ = self.get_indexed_on_node(inode)
            vs0.append(self.S[ivx])
        vx = BilinearInterpolation(x, y, x0, x1, y0, y1, vs0)
        return vx
