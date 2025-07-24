import quimb as qu
import quimb.tensor as qtn
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os
os.environ["NUMBA_NUM_THREADS"] = "1"

class HaldaneSpinHalf:
    
    #---------------------------------------------------------------------------------#
    def __init__(self, L, bond): 

        self.L = L # Number of particle
        self.bond = bond # Bond dimension
        self.X = qu.pauli('X').real
        self.Z = qu.pauli('Z').real
        self.I = qu.eye(2).real
    #---------------------------------------------------------------------------------#
    def MPO(self, h1, h2):

        W = np.zeros([5, 5, 2, 2], dtype = float)
        W[0, 0, :, :] = self.I
        W[0, 1, :, :] = self.X
        W[0, 2, :, :] = self.Z
        W[0, 4, :, :] = -h1 * self.X
        W[1, 4, :, :] = -h2 * self.X
        W[2, 3, :, :] = self.X
        W[3, 4, :, :] = -self.Z
        W[4, 4, :, :] = self.I
        Wl = W[0, :, :, :]
        Wr = W[:, 4, :, :]
        H = qtn.MatrixProductOperator([Wl] + [W] * (self.L - 2) + [Wr])

        return H
    #---------------------------------------------------------------------------------#
    def Projection(self):

        '''
        In this part we are making projection. The structure of code due to 
        reducing computational cost tune in a way that we can just consider
        odd number of sites.
        '''

        def make_projection(kind = 'W', sign = 1, side = False):

            if kind == 'W':
                base = self.Z
                dtype = float
            elif kind == 'I':
                base = self.I
                dtype = int
            else:
                raise ValueError('Invalid kind, use either "W" or "I"')

            if side: 
                tensor = np.zeros([2, 2, 2], dtype = dtype)
                tensor[0] = self.I
                tensor[1] = sign * base

            else:
                tensor = np.zeros([2, 2, 2, 2], dtype = dtype)
                tensor[0, 0] = self.I
                tensor[1, 1] = sign * base

            return tensor

        W = make_projection('W')
        Wr = make_projection('W', side=True)
        Wl = make_projection('W', sign=-1, side=True)

        Identity = make_projection('I')
        Identity_side_plus = make_projection('I', side=True)
        Identity_side_minus = make_projection('I', sign=-1, side=True)
        
        even_form = [Identity]+[W]
        even_repeat = sum([even_form for i in range(int((self.L-3)/2))],[])

        odd_form = [W]+[Identity]
        odd_repeat = sum([odd_form for i in range(int((self.L-3)/2))],[])

        P_plus_even = qtn.MatrixProductOperator([Wr] +  even_repeat + [Identity] + [Wr])
        P_plus_odd = qtn.MatrixProductOperator([Identity_side_plus] + odd_repeat + [W] + [Identity_side_plus])
        P_minus_even = qtn.MatrixProductOperator([Wl] + even_repeat+ [Identity] + [Wr])
        P_minus_odd = qtn.MatrixProductOperator([Identity_side_minus] + odd_repeat + [W] + [Identity_side_plus])

        return P_plus_even, P_plus_odd, P_minus_even, P_minus_odd
    #---------------------------------------------------------------------------------#
    def DMRG(self, h1, h2):
        DMRG = qtn.tensor_dmrg.DMRG(ham = self.MPO(h1 = h1, h2 = h2), bond_dims = self.bond) 
        DMRG.solve(tol = 1e-3, verbosity = 0)
        ground_state = DMRG.state
        return ground_state
    #---------------------------------------------------------------------------------#
    def generate_train_set(self, kind = 'random_points'):

        P_plus_even, P_plus_odd, P_minus_even, P_minus_odd = self.Projection()

        if kind == 'random_points':
            base = pd.read_csv('~/4th_project_with_UCL/supervised learning/Second_hamiltonian/dataset/random_train_set.csv')

        elif kind == 'regular_points':
            base = pd.read_csv('~/4th_project_with_UCL/supervised learning/Second_hamiltonian/dataset/regular_train_set.csv')

        else:
            raise ValueError('Invalid kind, use either "random_points" or "regular_points"')

        lst_project_state = []
        lst_project_target = []

        for i in tqdm(range(len(base)),desc=f'Generating train set for {kind}'):
            
            h1 = base['h1']
            h2 = base['h2']
            target = base['target']

            DMRG_state = self.DMRG(h1 = h1[i], h2 = h2[i])

            contraction_state_plus_even = P_plus_even.apply(DMRG_state); # projection state after P plus even projection
            normalisation_factor_plus_even = np.sqrt(np.abs(contraction_state_plus_even.H @ contraction_state_plus_even))
            if normalisation_factor_plus_even > 0.01:
                contraction_state_plus_even = contraction_state_plus_even/normalisation_factor_plus_even
                lst_project_state.append(contraction_state_plus_even)
                lst_project_target.append(target[i])

            contraction_state_plus_odd = P_plus_odd.apply(DMRG_state); # projection state after P plus odd projection
            normalisation_factor_plus_odd = np.sqrt(np.abs(contraction_state_plus_odd.H @ contraction_state_plus_odd))
            if normalisation_factor_plus_odd > 0.01:
                contraction_state_plus_odd = contraction_state_plus_odd/normalisation_factor_plus_odd
                lst_project_state.append(contraction_state_plus_odd)
                lst_project_target.append(target[i])

            contraction_state_minus_even = P_minus_even.apply(DMRG_state); # projection state after P minus even projection
            normalisation_factor_minus_even = np.sqrt(np.abs(contraction_state_minus_even.H @ contraction_state_minus_even))
            if normalisation_factor_minus_even > 0.01:
                contraction_state_minus_even = contraction_state_minus_even/normalisation_factor_minus_even
                lst_project_state.append(contraction_state_minus_even)
                lst_project_target.append(target[i])

            contraction_state_minus_odd = P_minus_odd.apply(DMRG_state); # projection state after P minus odd projection
            normalisation_factor_minus_odd = np.sqrt(np.abs(contraction_state_minus_odd.H @ contraction_state_minus_odd))
            if normalisation_factor_minus_odd > 0.01:
                contraction_state_minus_odd = contraction_state_minus_odd/normalisation_factor_minus_odd
                lst_project_state.append(contraction_state_minus_odd)
                lst_project_target.append(target[i])

        states = np.array(lst_project_state)
        targets = np.array(lst_project_target)

        return states, targets
    

    def generate_test_set(self):

        test_points = pd.read_csv('~/4th_project_with_UCL/supervised learning/Second_hamiltonian/dataset/test_set.csv')
        h1 = test_points['h1']
        h2 = test_points['h2']

        def compute_dmrg(i):
            return self.DMRG(h1 = h1[i], h2 = h2[i])
        
        results = Parallel(n_jobs=-1)(delayed(compute_dmrg)(i) for i  in tqdm(range(len(test_points)),desc = 'Generating test set'))
    
        return np.array(results)
