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
from Haldane_anis_supervise_class import *


class partial:
    
    def __init__(self, L, bond):
        self.L = L
        self.bond = bond
        self._train_dataset_path = None
        self._test_dataset_path = None

    def path(self, keep=None):
        path = f"./train_set_L={self.L}_partial_from_0_to_3_spins"
        os.makedirs(path, exist_ok=True)
        return path

    def generate_dataset(self, generate_test_set=True):

        print("Starting dataset generation — this step includes both the training and test sets. Please be patient...")

        start_time = time.time()
        points = Haldan_anis(L=self.L, bond=self.bond).points()
        file_path_train_DMRG = os.path.join(self.path(), f'train_set_DMRG_for_{self.L}_sites.pkl')
        with open(file_path_train_DMRG, "wb") as f:
            pickle.dump(points, f)
        print(f"Train_set is generated in {time.time() - start_time:.2f} seconds.")

        return points

    def partial_density_matrix(self, keep=None):
        print("Computing partial density matrices started .....")
        start_time = time.time()

        path = self.path(keep=keep)

        file_path_train_DMRG = os.path.join(path, f'train_set_DMRG_for_{self.L}_sites.pkl')
        with open(file_path_train_DMRG, "rb") as f:
            loaded_dataset = pickle.load(f)

        #loaded_dataset = self._load_dataset()[0]
        Xtr = loaded_dataset[0]
        d2 = len(Xtr)

        print(f"Tracing over training set ({d2} items)...")
        partial_rhos_train = Parallel(n_jobs=-1)(
            delayed(lambda x: x.partial_trace_to_dense_canonical(where=keep))(Xtr[i])
            for i in tqdm(range(d2), desc="Tracing train set"))

        print(f"Partial density matrices computed in {time.time() - start_time:.2f} seconds.")

        return partial_rhos_train


    def gram_train_partial(self,partial_train, keep=None):

        print("Computing Gram matrix for training set...")
        start_time = time.time()

        partial_rho = partial_train

        '''partial_rho = self.partial_density_matrix()[0]

        partial_rho = self._load_partial_density_matrix()[0]'''
        d = len(partial_rho)
        gram = np.zeros((d, d))
        
        for idx in tqdm(range(d * d), desc='Gram Partial Train'):
            i = idx // d
            j = idx % d
            if j >= i:
                gram[i, j] = gram[j, i] = (np.trace(partial_rho[i] @ partial_rho[j]).real)**2
        file_path_kernel_train_DMRG = os.path.join(self.path(keep=keep), f"kernel_train_Haldane_DMRG_partial_from_{keep[0]}_to_{keep[-1]}_spins.hdf5")
        with h5py.File(file_path_kernel_train_DMRG, "w") as f:
            f.create_dataset("gram_train_DMRG_partial", data=gram)

        print(f"Gram matrix for training set computed in {time.time() - start_time:.2f} seconds.")
        return gram
