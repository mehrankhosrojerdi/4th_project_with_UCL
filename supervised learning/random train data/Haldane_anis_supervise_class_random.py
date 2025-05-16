import quimb as qu
import quimb.tensor as qtn
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd


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
        dmrg_solver.solve(tol = 1e-3, verbosity = 0)
        ground_state = dmrg_solver.state
        return ground_state
    #-----------------------------------------------------------------------#
    def points(self):

        df = pd.read_csv('random_train_dataset.csv')

        lst_x = df['x'].tolist()
        lst_y = df['y'].tolist()
        lst_target = df['target'].tolist()

        def compute_dmrg(d, e):
            return self.DMRG(d1=d, e1=e)

        lst_DMRG = Parallel(n_jobs=-1, backend = 'loky')(delayed(compute_dmrg)(x, y) for x,y in tqdm(zip(lst_x, lst_y),desc='Generating train set'))

        return lst_DMRG, lst_target

  
    def generate_test_set(self):
        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)
        all_points = [(d, e) for e in E for d in D]
        def compute_dmrg(d, e):
            return self.DMRG(d1=d, e1=e)

        results = Parallel(n_jobs=-1, backend = 'loky')(delayed(compute_dmrg)(d, e) for (d, e) in tqdm(all_points,desc='Generating test set' ))

        
        return results
