import quimb as qu
import quimb.tensor as qtn
import numpy as np
import time
from joblib import Parallel, delayed
from tqdm import tqdm

class TransferLearning:
    '''
    This class implements the transfer learning workflow.
    It provides functions to convert both the Haldane 
    anisotropic model and the XY model into Matrix Product 
    Operators (MPOs), and to compute optimized ground states 
    using the DMRG algorithm. It also includes functions for 
    generating training and test datasets, along with their 
    corresponding target labels, and for constructing kernel 
    matrices for both sets.
    '''
    def __init__(self, L, bond):
        self.L = L         # number of particle
        self.bond = bond   # bond dimension

    def Haldane_MPO(self, D, E):
        '''
        Constructs the Matrix Product Operator 
        (MPO) for the Haldane anisotropic model.
        '''
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

        MPO_Haldane = qtn.MatrixProductOperator([Wl] + [W] * (self.L - 2) + [Wr])

        return MPO_Haldane
    
    def XY_MPO(self, gamma, bz_value):
        '''
        Constructs the Matrix Product Operator 
        (MPO) for the XY model.
        '''
        MPO_XY = qtn.MPO_ham_XY(self.L, j=((1+gamma),(1-gamma)), bz=bz_value, S=1)
        
        return MPO_XY
    
    def dmrg_solver(self, H):
        """
        Compute the ground state of a given Hamiltonian MPO using DMRG.

        Parameters:
            H (qtn.MatrixProductOperator): Hamiltonian MPO.

        Returns:
            qtn.MPS: The optimized ground-state MPS.
        """
        DMRG = qtn.tensor_dmrg.DMRG(ham = H, bond_dims = self.bond, cutoffs=1e-12) 
        DMRG.solve(tol = 1e-6, verbosity = 0)
        ground_state = DMRG.state
        return ground_state
    
    def generate_trainset(self):
        '''
        Generates the training dataset for the 
        Haldane anisotropic model.
        '''

        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)
        lst_points = []
        lst_target = []

        for e in E:
            if 0.8 < e < 2.0:
                lst_points.append([-2, e])
                lst_target.append(1)  # 'large_ex'
            elif -0.8 < e < 0.8:
                lst_points.append([-2, e])
                lst_target.append(2)  # 'z_neel'
            elif -2 < e < -0.8:
                lst_points.append([-2, e])
                lst_target.append(3)  # 'large_ey'

        for e in E:
            if -2 < e < -0.4:
                lst_points.append([2, e])
                lst_target.append(6)  # 'x_neel'
            elif -0.4 < e < 0.4:
                lst_points.append([2, e])
                lst_target.append(5)  # 'large_d'
            elif 0.4 < e < 2.0:
                lst_points.append([2.0, e])
                lst_target.append(4)  # 'y_neel'

        for d in D:
            if -2 < d < 0.2:
                lst_points.append([d, 2])
                lst_target.append(1)  # 'large_ex'
            elif 0.2 < d < 2.0:
                lst_points.append([d, 2.0])
                lst_target.append(4)  # 'y_neel'

        for d in D:
            if -2 < d < 0.2:
                lst_points.append([d, -2])
                lst_target.append(3)  # 'large_ey'
            elif 0.2 < d < 2.0:
                lst_points.append([d, -2])
                lst_target.append(6)  # 'x_neel'

        for d in np.arange(-2, -1, 0.1):  
            lst_points.append([d, 0.0]) # 'z_neel'
            lst_target.append(2) 

        for d in np.arange(1.1, 2, 0.1):
            lst_points.append([d, 0.0])
            lst_target.append(5)  # 'large_d'

        for d in np.arange(-0.4, 0.6, 0.1):
            lst_points.append([d, 0.0])
            lst_target.append(0) #'Haldane'
        def compute_dmrg(d, e):
            MPO = self.Haldane_MPO(d, e)
            return self.dmrg_solver(MPO)

        lst_DMRG = Parallel(n_jobs=-1, backend = 'loky')(delayed(compute_dmrg)(point[0], point[1]) for point in tqdm(lst_points,desc='Generating train set'))

        return lst_DMRG, lst_target
    
    def generate_testset(self):
        '''
        Generates the test dataset for the 
        XY model.
        '''
        gamma_value = np.linspace((1e-5), 1, 20)
        bz_value = np.linspace((1e-5), 1, 20)

        def compute_dmrg(gamma, bz):
            MPO = self.XY_MPO(gamma, bz)
            return self.dmrg_solver(MPO)
        
        lst_test = []
        for gamma in gamma_value:
            for bz in bz_value:
                lst_test.append([gamma, bz])

        lst_DMRG = Parallel(n_jobs=-1, backend = 'loky')(delayed(compute_dmrg)(res[0], res[1]) for res in tqdm(lst_test,desc='Generating test set'))
        
        return lst_DMRG
    
    def get_kernel_train(self):
        '''
        This function computes the kernel matrix for the training set

        parameters:
            train: list of MPS for the training set

        returns:
            kernel_train: kernel matrix for the training set
        '''
        print("Computing Gram matrix for training set...")
        start_time = time.time()

        psi = self.generate_trainset()[0]

        d = len(psi)
        gram = np.zeros((d, d))
        
        for idx in tqdm(range(d * d), desc='Gram Train'):
            i = idx // d
            j = idx % d
            if j >= i:
                gram[i, j] = gram[j, i] = ((psi[i].H @ psi[j]).real)**2

        print(f"Gram matrix for training set computed in {time.time() - start_time:.2f} seconds.")
        return gram
    
    def get_kernel_test(self):
        '''
        This function computes the kernel matrix for the test set

        parameters:
            test: list of MPS for the test and train set

        returns:
            kernel_test: kernel matrix for the test set
        '''
        print("Computing Gram matrix for test set...")
        start_time = time.time()

        psi_train = self.generate_trainset()[0]
        psi_test = self.generate_testset()    

        d1 = len(psi_test)
        d2 = len(psi_train)
        gram_matrix_test = np.zeros((d1,d2))
        for i in tqdm(range(d1), desc='Gram Test'):
            for j in range(d2):
                gram_matrix_test[i,j] = ((psi_test[i].H @ psi_train[j]).real)**2
        print(f"Gram matrix for testing set computed in {time.time() - start_time:.2f} seconds.")
        return gram_matrix_test   
