import quimb as qu
import quimb.tensor as qtn
import numpy as np
import time
import os
import pickle
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm


class Haldan_anis:

    #-----------------------------------------------------------------------#
    def __init__(self, L, bond):
        self.L = L 
        self.bond =bond
    #-----------------------------------------------------------------------#
    def MPO(self, D, E):

        J=1
        I = qu.eye(3).real  
        Sx = qu.spin_operator('X', S=1).real
        Sy = qu.spin_operator('Y', S=1)
        Sz = qu.spin_operator('Z', S=1).real
        W = np.zeros([5, 5, 3, 3], dtype=complex)

        W[0, 0, :, :] = I  
        W[0, 1, :, :] = J * Sx
        W[0, 2, :, :] = J * Sy
        W[0, 3, :, :] = J * Sz
        W[0, 4, :, :] = D * (Sz @ Sz) + E * (Sx @ Sx - Sy @ Sy)

        W[1, 4, :, :] = Sx
        W[2, 4, :, :] = Sy
        W[3, 4, :, :] = Sz
        W[4, 4, :, :] = I  

        Wl = W[0, :, :, :]  
        Wr = W[:, 4, :, :]  

        H = qtn.MatrixProductOperator([Wl] + [W] * (self.L - 2) + [Wr])

        return H
    #-----------------------------------------------------------------------#
    def DMRG(self, d1, e1):
        dmrg_solver = qtn.tensor_dmrg.DMRG(ham = self.MPO(D = d1, E = e1), bond_dims = self.bond) 
        dmrg_solver.solve(tol = 1e-3, verbosity = 0);
        ground_state = dmrg_solver.state
        return ground_state
    #-----------------------------------------------------------------------#
    def points(self):
        lst_points = [[-0.6,-0.35],[-0.35,-0.6],[-0.25,-0.25],[-0.75,-0.75]]
        lst_target = [1,2,3,4]
        def compute_dmrg(d, e):
            return self.DMRG(d1=d, e1=e)

        lst_DMRG = Parallel(n_jobs=-1, backend = 'loky')(delayed(compute_dmrg)(point[0], point[1]) for point in tqdm(lst_points,desc='Generating train set'))

        return lst_DMRG, lst_target



  


    
    