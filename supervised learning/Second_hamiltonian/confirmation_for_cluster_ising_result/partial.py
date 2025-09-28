import os
os.environ["NUMBA_NUM_THREADS"] = "1" 
from tqdm import tqdm
import pickle
import h5py
import time
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from joblib import Parallel, delayed
from Haldane_spin_half import *

class partial:
    
    def __init__(self, L, bond):
        self.L = L
        self.bond = bond


    def path(self):
        path = f"./train_and_testset_for_L={self.L}_Bond={self.bond}"
        os.makedirs(path, exist_ok=True)
        return path
    
    def generate_dataset(self, generate_train_set = True, generate_test_set = True):

        print("Starting dataset generation — this step includes both the training and test sets. Please be patient...")

        model = HaldaneSpinHalf(L=self.L, bond=self.bond)

        if generate_train_set:
            start_time = time.time()
            train = model.generate_train_set(kind = 'random_points')
            file_path_train_DMRG = os.path.join(self.path(), f'train_set_DMRG and tragets.pkl')
            with open(file_path_train_DMRG, "wb") as f:
                pickle.dump(train, f)
            print(f"Train_set is generated in {time.time() - start_time:.2f} seconds.")

        if generate_test_set:
            start_time  = time.time()
            test = model.generate_test_set()
            file_path_test_DMRG = os.path.join(self.path(), f'test_set_DMRG.pkl')
            with open(file_path_test_DMRG, "wb") as f:
                pickle.dump(test, f)
            print(f"Test_set is generated in {time.time() - start_time:.2f} seconds.")

        

    def partial_density_matrix(self, keep=None):
        print("Computing partial density matrices started .....")
        start_time = time.time()

        file_path_train_DMRG = os.path.join(self.path(), f'train_set_DMRG and tragets.pkl')
        with open(file_path_train_DMRG, "rb") as f:
            loaded_train_set = pickle.load(f)

        file_path_test_DMRG = os.path.join(self.path(), f'test_set_DMRG.pkl')
        with open(file_path_test_DMRG, "rb") as f:
            loaded_test_set = pickle.load(f)

        Xte = loaded_test_set
        d1 = len(Xte)

        Xtr = loaded_train_set[0]
        d2 = len(Xtr)

        print(f"Tracing over training set ({d2} items)...")
        partial_rhos_train = Parallel(n_jobs=-1)(
            delayed(lambda x: x.partial_trace_to_dense_canonical(where=keep))(Xtr[i])
            for i in tqdm(range(d2), desc="Tracing train set"))

        print(f"Tracing over test set ({d1} items)...")    
        partial_rhos_test = Parallel(n_jobs=-1)(
            delayed(lambda x: x.partial_trace_to_dense_canonical(where=keep))(Xte[i])
            for i in tqdm(range(d1), desc="Tracing test set"))        

        print(f"Partial density matrices computed in {time.time() - start_time:.2f} seconds.")

        return partial_rhos_train, partial_rhos_test

    def gram_train_partial(self,partial_train, keep=None):

        print("Computing Gram matrix for training set...")
        start_time = time.time()

        partial_rho = partial_train

        d = len(partial_rho)
        gram = np.zeros((d, d))
        
        for idx in tqdm(range(d * d), desc='Gram Partial Train'):
            i = idx // d
            j = idx % d
            if j >= i:
                gram[i, j] = gram[j, i] = (np.trace(partial_rho[i] @ partial_rho[j]).real)**2
        file_path_kernel = os.path.join(self.path(), f"kernel_partial_keep_from_{keep[0]}_to_{keep[-1]}")
        os.makedirs(file_path_kernel, exist_ok=True)
        file_path_kernel_train_DMRG = os.path.join(file_path_kernel,f"kernel_train_Haldane_DMRG_partial_from_{keep[0]}_to_{keep[-1]}_spins.hdf5")
        with h5py.File(file_path_kernel_train_DMRG, "w") as f:
            f.create_dataset("gram_train_DMRG_partial", data=gram)

        print(f"Gram matrix for training set computed in {time.time() - start_time:.2f} seconds.")
        return gram

    def gram_test_partial(self, partial_train, partial_test, keep=None):

        print("Computing Gram matrix for testing set...")
        start_time = time.time()

        partial_rhos_train = partial_train
        partial_rhos_test = partial_test    

        d1 = len(partial_rhos_test)
        d2 = len(partial_rhos_train)
        gram_matrix_test = np.zeros((d1,d2))
        for i in tqdm(range(d1), desc='Gram Partial Test'):
            for j in range(d2):
                gram_matrix_test[i,j] = (np.trace(partial_rhos_test[i] @ partial_rhos_train[j]).real)**2
        
        file_path_kernel = os.path.join(self.path(), f"kernel_partial_keep_from_{keep[0]}_to_{keep[-1]}")
        os.makedirs(file_path_kernel, exist_ok=True)
        file_path_kernel_test_DMRG = os.path.join(file_path_kernel, f"kernel_test_Haldane_DMRG_partial_from_{keep[0]}_to_{keep[-1]}_spins.hdf5")
        with h5py.File(file_path_kernel_test_DMRG, "w") as f:
            f.create_dataset("gram_test_DMRG_partial", data = gram_matrix_test)
        print(f"Gram matrix for testing set computed in {time.time() - start_time:.2f} seconds.")
        return gram_matrix_test   
