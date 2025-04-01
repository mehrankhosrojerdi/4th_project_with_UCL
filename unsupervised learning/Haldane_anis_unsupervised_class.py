import quimb as qu
import quimb.tensor as qtn
import numpy as np
from joblib import Parallel, delayed


class Haldan_anis_unsupervised:

    #-----------------------------------------------------------------------#
    def __init__(self, L, ls, bond):
        self.L = L 
        self.ls = ls 
        self.bond = bond
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
        DMRG = qtn.tensor_dmrg.DMRG(ham = Haldan_anis_unsupervised(L = self.L, ls = self.ls, bond = self.bond).MPO(D = d1, E = e1), bond_dims = self.bond) 
        DMRG.solve(tol = 1e-8, verbosity = 0);
        ground_state = DMRG.state
        return ground_state
    #-----------------------------------------------------------------------#
    def generate_Entire_set(self):

        D = np.linspace(-2, 2, int(self.ls))
        E = np.linspace(-2, 2, int(self.ls))
        
        result = np.array(np.meshgrid(D, E)).T.reshape(-1, 2)
        lst_DMRG_state = Parallel(n_jobs=5, backend='loky')(delayed(Haldan_anis_unsupervised(L = self.L, ls = self.ls, bond = self.bond).DMRG)(d, e) for d, e in result)


        return np.array(lst_DMRG_state)