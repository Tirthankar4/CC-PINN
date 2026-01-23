from core.data_generator import col_gen
from core.data_generator import diff

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from config import cs, const, G, rho_o

class ASTPN(col_gen):
    
    def __init__(self, rmin=[0,0,0,0], rmax=[1,1,1,1], N_0 = 1000, N_b=1000, N_r=3000, dimension=1):
        super().__init__(rmin,rmax, N_0,0,N_r, dimension)  # N_b set to 0 due to hard constraints
        
       
        self.coord_Lx, self.coord_Rx = self.geo_time_coord(option="BC",coordinate=1)
        
        if dimension == 2:
            self.coord_Ly, self.coord_Ry = self.geo_time_coord(option="BC",coordinate=2)

        if dimension == 3:
            self.coord_Ly, self.coord_Ry = self.geo_time_coord(option="BC",coordinate=2)
            self.coord_Lz, self.coord_Rz = self.geo_time_coord(option="BC",coordinate=3)


def pde_residue(colloc, net, dimension = 1):
    
    '''
    This is the main function that returns all the PDE residue
    
    Args:
        colloc: Collocation points
        net: Neural network
        dimension: Spatial dimension (1, 2, or 3)
    '''
    
    return pde_residue_standard(colloc, net, dimension)


def pde_residue_standard(colloc, net, dimension = 1):
    
    '''
    Standard PDE residues (network predicts rho directly)
    '''
    net_outputs = net(colloc)
    
    x = colloc[0]
    
    if dimension == 1:
        t = colloc[1]

    elif dimension == 2:
        y = colloc[1]
        t = colloc[2]

    elif dimension == 3:
        y = colloc[1]
        z = colloc[2]
        t = colloc[3]
    
    rho, vx = net_outputs[:,0:1], net_outputs[:,1:2]

    if dimension == 1:

        phi = net_outputs[:,2:3]

        rho_t = diff(rho,t,order=1)  
        rho_x = diff(rho,x,order=1)

        vx_t = diff(vx, t,order=1)
        vx_x = diff(vx, x,order=1)
        
        phi_x = diff(phi,x,order=1)
        phi_x_x = diff(phi,x,order=2)

    elif dimension == 2:

        vy = net_outputs[:,2:3]
        phi = net_outputs[:,3:4]

        rho_t = diff(rho,t,order=1)  
        rho_x = diff(rho,x,order=1)
        rho_y = diff(rho,y,order=1)

        vx_t = diff(vx, t,order=1)
        vy_t = diff(vy, t,order=1)

        vx_x = diff(vx, x,order=1)
        vx_y = diff(vx, y,order=1)
        vy_x = diff(vy, x,order=1)
        vy_y = diff(vy, y,order=1)
        
        phi_x = diff(phi,x,order=1)
        phi_x_x = diff(phi,x,order=2)

        phi_y = diff(phi,y,order=1)
        phi_y_y = diff(phi,y,order=2)

    elif dimension == 3:
        vy = net_outputs[:,2:3]
        vz = net_outputs[:,3:4]
        phi = net_outputs[:,4:5]

        rho_t = diff(rho,t,order=1)  
        rho_x = diff(rho,x,order=1)
        rho_y = diff(rho,y,order=1)
        rho_z = diff(rho,z,order=1)

        vx_t = diff(vx, t,order=1)
        vy_t = diff(vy, t,order=1)
        vz_t = diff(vz, t,order=1)

        vx_x = diff(vx, x,order=1)
        vy_x = diff(vy, x,order=1)
        vz_x = diff(vz, x,order=1)

        vx_y = diff(vx, y,order=1)
        vy_y = diff(vy, y,order=1)
        vz_y = diff(vz, y,order=1)
        
        vx_z = diff(vx, z,order=1)
        vy_z = diff(vy, z,order=1)
        vz_z = diff(vz, z,order=1)
        
        phi_x = diff(phi,x,order=1)
        phi_x_x = diff(phi,x,order=2)

        phi_y = diff(phi,y,order=1)
        phi_y_y = diff(phi,y,order=2)
    
        phi_z = diff(phi,z,order=1)
        phi_z_z = diff(phi,z,order=2)

    
    ## The residues from the equations

    if dimension == 1:
        rho_r = rho_t + vx * rho_x + rho * vx_x
        vx_r = rho*vx_t + rho*(vx*vx_x) + cs*cs*rho_x +rho*phi_x
        phi_r = phi_x_x - const*(rho - rho_o)

        return rho_r, vx_r, phi_r

    elif dimension == 2:
        rho_r = rho_t + vx * rho_x + vy * rho_y + rho * vx_x + rho * vy_y
        vx_r = rho*vx_t + rho*(vx*vx_x + vy*vx_y) + cs*cs*rho_x + rho*phi_x
        vy_r = rho*vy_t + rho*(vy*vy_y + vx*vy_x) + cs*cs*rho_y + rho*phi_y
        phi_r = phi_x_x + phi_y_y - const*(rho - rho_o)

        return rho_r, vx_r, vy_r, phi_r
    
    elif dimension == 3:
        rho_r = rho_t + vx * rho_x + rho * vx_x + vy *rho_y + rho * vy_y + vz *rho_z +rho * vz_z
        vx_r = rho*vx_t + rho*(vx*vx_x + vy*vx_y+vz*vx_z) + cs*cs*rho_x + rho*phi_x
        vy_r = rho*vy_t + rho*(vy*vy_y + vx*vy_x+vz*vy_z) + cs*cs*rho_y + rho*phi_y
        vz_r = rho*vz_t + rho*(vz*vz_z + vx*vz_x+vy*vz_y) + cs*cs*rho_z + rho*phi_z
        phi_r = phi_x_x + phi_y_y +phi_z_z - const*(rho - rho_o)
        
        return rho_r,vx_r,vy_r,vz_r,phi_r


def poisson_residue_only(colloc, net, dimension=1):
    """
    Compute only the Poisson equation residual: ∇²φ - const*(ρ - ρ₀)
    
    This is used for extra enforcement at t=0 (Option 3: Pure ML approach).
    By evaluating Poisson residual on many t=0 points, we ensure φ is 
    correctly initialized without using numerical solvers.
    
    Args:
        colloc: Collocation points [x, (y), (z), t]
        net: Neural network
        dimension: Spatial dimension (1, 2, or 3)
    
    Returns:
        phi_r: Poisson residual tensor
    """
    net_outputs = net(colloc)
    
    x = colloc[0]
    
    if dimension == 1:
        t = colloc[1]
        phi = net_outputs[:, 2:3]
        phi_x_x = diff(phi, x, order=2)
        
    elif dimension == 2:
        y = colloc[1]
        t = colloc[2]
        phi = net_outputs[:, 3:4]
        phi_x_x = diff(phi, x, order=2)
        phi_y_y = diff(phi, y, order=2)
        
    elif dimension == 3:
        y = colloc[1]
        z = colloc[2]
        t = colloc[3]
        phi = net_outputs[:, 4:5]
        phi_x_x = diff(phi, x, order=2)
        phi_y_y = diff(phi, y, order=2)
        phi_z_z = diff(phi, z, order=2)
    
    # Get density
    rho = net_outputs[:, 0:1]
    
    # Compute Poisson residual
    if dimension == 1:
        phi_r = phi_x_x - const*(rho - rho_o)
    elif dimension == 2:
        phi_r = phi_x_x + phi_y_y - const*(rho - rho_o)
    elif dimension == 3:
        phi_r = phi_x_x + phi_y_y + phi_z_z - const*(rho - rho_o)
    
    return phi_r