'''
This file defines a MESH2D class.
'''

import numpy as np

class MESH2D():
    '''
    class for a 2-d mesh
    Attributes:
        xs (1-d ndarray): x coordinates
        ys (1-d ndarray): y coordinates
        xnum (int): number of points along x
        ynum (int): number of points along y
        nnode (int): number of points in the domain, including ghost points
        is_uniform (bool): if the mesh is uniform
    '''
    def __init__(self, xs, ys, **kwargs):
        '''
        Initiation
        Inputs:
            xs (1-d ndarray): x coordinates
            ys (1-d ndarray): y coordinates
            kwargs (dict):
                is_uniform (bool): if the mesh is uniform
        '''
        self.xs = xs
        self.ys = ys
        self.xnum = xs.size
        self.ynum = ys.size
        self.nnode = (self.xnum+1) * (self.ynum+1) # ghost points included
        self.is_uniform = kwargs.get('is uniform', True)
        pass

    def get_node_index(self, iy, jx):
        '''
        get the index for a node from index along x, y
        Inputs:
            iy (int): y index
            jx (int): x index
        Returns:
            inode (int): index of node
        '''
        inode = jx * (self.ynum + 1) + iy
        return inode
    
    def get_node_center(self, inode):
        '''
        Get the coordinates of node center
        '''
        iy, jx = self.get_indexes_of_node(inode)
        if iy == self.ys.size:
            y_center = self.ys[-1] + (self.ys[-1] - self.ys[-2]) / 2.0
        elif iy == 0:
            y_center = self.ys[0] / 2.0
        else:
            y_center = (self.ys[iy] + self.ys[iy-1]) / 2.0
        if jx == self.xs.size:
            x_center = self.xs[-1] + (self.xs[-1] - self.xs[-2]) / 2.0
        elif jx == 0:
            x_center =self.xs[0] / 2.0
        else:
            x_center = (self.xs[jx] + self.xs[jx-1]) / 2.0
        return x_center, y_center
    
    def get_node_size(self, x, y):
        '''
        Get the size of the node
        Inputs:
            x, y: coordinates of a point
        Returns:
            x_stp, y_stp: size of the nodes
        '''
        iy = self.iy_up_to(y)
        jx = self.jx_left_to(x)
        x_stp = self.xs[jx+1] - self.xs[jx]
        y_stp = self.ys[iy+1] - self.ys[iy]
        return x_stp, y_stp

    def get_indexes_of_node(self, inode):
        '''
        get the indexes along x, y from the index of a node
        Inputs:
            inode (int): node index
        Returns:
            iy, jx (index of nodes)
        '''
        jx = inode // (self.ynum + 1)
        iy = inode % (self.ynum + 1)
        return iy, jx
        
    def get_indexed_on_node(self, inode):
        '''
        indexing for P, vx, vy nodes from an index of a node
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

    def get_coordinates(self):
        '''
        return coordinates of x and y
        Returns:
            xs (1-d ndarray): x coordinates
            ys (1-d ndarray): y coordinates
        '''
        return self.xs, self.ys
    
    def get_sizes(self):
        '''
        Return sizes of the domain in x and y,
        assuming, xs[0] = 0.0 and ys[0] = 0.0
        Returns:
            x_size (float): x size
            y_size (float): y size
        '''
        return self.xs[-1], self.ys[-1]
    
    def get_number_of_points(self):
        '''
        Return number of mesh points
        '''
        return self.nnode
    
    def get_number_of_points_xy(self):
        '''
        Return number of mesh points along x, y respectivly
        '''
        return self.xs.size, self.ys.size
    
    def get_vy_point(self, iy, jx):
        '''
        Return the coordinates of a single vy node
        Inputs:
            iy (int): index along y
            ix (int): index along x
        Returns:
            x (float): x coordinate
            y (float): y coordinate
        '''
        if jx == 0:
            x = self.xs[0] - (self.xs[1] - self.xs[0]) / 2.0
        elif jx == self.xs.size:
            x = self.xs[-1] + (self.xs[-1] - self.xs[-2]) / 2.0
        else:
            x = (self.xs[jx] + self.xs[jx-1]) / 2.0
        y = self.ys[iy]
        return x, y

    def get_vx_point(self, iy, jx):
        '''
        Return the coordinates of a single vx node
        Inputs:
            iy (int): index along y
            ix (int): index along x
        Returns:
            x (float): x coordinate
            y (float): y coordinate
        '''
        x = self.xs[jx]
        if iy==0:
            y = self.ys[0] - (self.ys[1] - self.ys[0]) / 2.0
        elif iy==self.ys.size:
            y = self.ys[-1] + (self.ys[-1] - self.ys[-2]) / 2.0
        else:
            y = (self.ys[iy] + self.ys[iy-1]) / 2.0
        return x, y
    
    def iy_up_to(self, y):
        '''
        get the y index above a y coordinate.
        The implementation of this function considers whether
        the mesh is uniform. If so, it will perform a single deviding operation.
        Otherwise, it will perform a bisection operation.
        Inputs:
            y (float): y axis entry
        '''
        ys = self.ys
        assert(ys[0] <= y and y <= ys[-1])
        ynum = ys.size
        if not self.is_uniform:
            id0 = 0  # non-uniform, use a bisection method
            id1 = ynum-1
            while (id1 - id0 > 1):
                id_mid = int((id0 + id1) / 2.0)
                y_mid = ys[id_mid]
                if y_mid < y:
                    id0 = id_mid
                else:
                    id1 = id_mid
            iy = id0
        else:
            ystp = ys[1] - ys[0]  # uniform
            iy = int(y/ystp)
        if y == ys[-1]:  # make an exception at the boundary
            iy = self.ynum - 2
        return iy

    def iy_up_to_plus_half(self, y):
        '''
        get the y index above a y coordinate and then migrate downward by 0.5.
        The implementation of this function considers whether
        the mesh is uniform. If so, it will perform a single deviding operation.
        Otherwise, it will perform a bisection operation.
        Inputs:
            y (float): y axis entry
        '''
        ys = self.ys
        ynum = ys.size
        if not self.is_uniform:
            id0 = 0  # non-uniform, use a bisection method
            id1 = ynum-1
            while (id1 - id0 > 1):
                id_mid = int((id0 + id1) / 2.0)
                y_mid = ys[id_mid]
                y_step = ys[id_mid+1] - ys[id_mid]
                if y_mid < y + y_step / 2.0:
                    id0 = id_mid
                else:
                    id1 = id_mid
            iy = id0
        else:
            ystp = ys[1] - ys[0]  # uniform
            iy = int(y/ystp + 0.5)
        return iy
    
    def jx_left_to(self, x):
        '''
        get the x index left to a x coordinate.
        The implementation of this function considers whether
        the mesh is uniform. If so, it will perform a single deviding operation.
        Otherwise, it will perform a bisection operation.
        Inputs:
            x (float): x axis entry
        Returns:
            jx (int): x index
        Notes:
            jx should be in the range of 0 and xnum - 2, it cannot be the rightmost point.
            Similar for iy in the counter function, which should be 0 - (ynum-2)
        '''
        xs = self.xs
        assert(xs[0] <= x and x <= xs[-1])
        xnum = xs.size
        if not self.is_uniform:
            id0 = 0  # get indexing of x
            id1 = xnum-1
            while (id1 - id0 > 1):
                id_mid = int((id0 + id1) / 2.0)
                x_mid = xs[id_mid]
                if x_mid < x:
                    id0 = id_mid
                else:
                    id1 = id_mid
            jx = id0  
        else:
            xstp = xs[1] - xs[0]
            jx = int(x/xstp)
        if x == xs[-1]:  # make this an exeption
            jx = xnum - 2
        return jx

    def jx_left_to_plus_half(self, x):
        '''
        get the x index left to a x coordinate and then migrate 0.5 to the right
        The implementation of this function considers whether
        the mesh is uniform. If so, it will perform a single deviding operation.
        Otherwise, it will perform a bisection operation.
        '''
        xs = self.xs
        xnum = xs.size
        if not self.is_uniform:
            id0 = 0  # get indexing of x
            id1 = xnum-1
            while (id1 - id0 > 1):
                id_mid = int((id0 + id1) / 2.0)
                x_mid = xs[id_mid]
                x_stp = xs[id_mid+1] - xs[id_mid] # node x length
                if x_mid < x + x_stp / 2.0:
                    id0 = id_mid
                else:
                    id1 = id_mid
            jx = id0  
        else:
            xstp = xs[1] - xs[0]
            jx = int(x/xstp + 0.5)
        return jx
    
    def node_index(self, x, y):
        '''
        Get the nearest nodes to a point x (i.e. corners of the cell it belongs to)
        Inputs:
            x, y: x and y coordinates
        Returns:
            inode0: index of the node locates at the upper-left corner of the cell
            inodeB: index of the node locates at the lower-left corner of the cell
            inodeBR: index of the node locates at the lower-left corner of the cell
            inodeR: index of the node locates at the upper-right corner of the cell
        '''
        iy = self.iy_up_to(y)
        jx = self.jx_left_to(x)
        inode0 = self.get_node_index(iy, jx)
        inodeB = self.get_node_index(iy+1, jx)
        inodeR = self.get_node_index(iy, jx+1)
        inodeBR =self.get_node_index(iy+1, jx+1)
        return inode0, inodeR, inodeBR, inodeB

    def get_cell_size(self, x, y):
        '''
        Get the size of a cell that contains a point (x, y)
        Inputs:
            x, y: coordinates of a point
        Returns:
            x_stp, y_stp: size(length and width) of the cell
        '''
        iy = self.iy_up_to(y)
        jx = self.jx_left_to(x)
        x_stp = self.xs[jx+1] - self.xs[jx]
        y_stp = self.ys[iy+1] - self.ys[iy]
        return x_stp, y_stp
    
    def get_vy_cell_nodes(self, x, y):
        '''
        get the index of the node locating at the upper-left corner of a vy cell
        x, y (float): coordinates
        Returns:
            xvys : x coordinates
            yvys : y coordinates
            inodes: node indexes
        '''
        inodes = []
        xvys = []
        yvys = []
        assert(self.is_in_mesh(x, y))
        iy = self.iy_up_to(y)
        jx = self.jx_left_to_plus_half(x)
        xvy, yvy = self.get_vy_point(iy, jx) # node 0
        inode0 = self.get_node_index(iy, jx)
        xvys.append(xvy)
        yvys.append(yvy)
        inodes.append(inode0)
        xvy, yvy = self.get_vy_point(iy, jx+1) # node 1
        inode0 = self.get_node_index(iy, jx+1)
        xvys.append(xvy)
        yvys.append(yvy)
        inodes.append(inode0)
        xvy, yvy = self.get_vy_point(iy+1, jx+1) # node 2
        inode0 = self.get_node_index(iy+1, jx+1)
        xvys.append(xvy)
        yvys.append(yvy)
        inodes.append(inode0)        
        xvy, yvy = self.get_vy_point(iy+1, jx) # node 3
        inode0 = self.get_node_index(iy+1, jx)
        xvys.append(xvy)
        yvys.append(yvy)
        inodes.append(inode0)
        return np.array(xvys), np.array(yvys), np.array(inodes)

    def get_vx_cell_nodes(self, x, y):
        '''
        get the index of the node locating at the upper-left corner of a vx cell
        x, y (float): coordinates
        Return:
            xvys : x coordinates
            yvys : y coordinates
            inodes: node indexes
        '''
        inodes = []
        xvxs = []
        yvxs = []
        assert(self.is_in_mesh(x, y))
        iy = self.iy_up_to_plus_half(y)
        jx = self.jx_left_to(x)
        xvx, yvx = self.get_vx_point(iy, jx) # node 0
        inode0 = self.get_node_index(iy, jx)
        xvxs.append(xvx)
        yvxs.append(yvx)
        inodes.append(inode0)
        xvx, yvx = self.get_vx_point(iy, jx+1) # node 1
        inode0 = self.get_node_index(iy, jx+1)
        xvxs.append(xvx)
        yvxs.append(yvx)
        inodes.append(inode0)
        xvx, yvx = self.get_vx_point(iy+1, jx+1) # node 2
        inode0 = self.get_node_index(iy+1, jx+1)
        xvxs.append(xvx)
        yvxs.append(yvx)
        inodes.append(inode0)
        xvx, yvx = self.get_vx_point(iy+1, jx) # node 3
        inode0 = self.get_node_index(iy+1, jx)
        xvxs.append(xvx)
        yvxs.append(yvx)
        inodes.append(inode0)
        return np.array(xvxs), np.array(yvxs), np.array(inodes)

    def is_in_mesh(self, x, y):
        '''
        determine whether x and y is in the mesh domain
        '''
        return (x>= self.xs[0] and x<= self.xs[-1] and y >= self.ys[0] and y <= self.ys[-1])
