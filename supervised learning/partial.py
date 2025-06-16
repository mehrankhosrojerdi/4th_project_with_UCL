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
    
    def __init__(self, L, keep, bond):
        self.L = L
        self.keep = keep
        self.bond = bond
        self._train_dataset_path = None
        self._test_dataset_path = None

    def path(self):
        k = self.keep
        path = f"./dataset_L=51_bond=50_partial"
        os.makedirs(path, exist_ok=True)
        return path

    def generate_dataset(self):

        print("Starting dataset generation — this step includes both the training and test sets. Please be patient...")

        start_time = time.time()
        # Generate and save training set
        points = Haldan_anis(L=self.L, bond=self.bond, keep=self.keep).points()
        file_path_train_DMRG = os.path.join(self.path(), f'train_set_DMRG_partial_{len(self.keep)}spins.pkl')
        with open(file_path_train_DMRG, "wb") as f:
            pickle.dump(points, f)
        self._train_dataset_path = file_path_train_DMRG
        print(f"Train_set is generated in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        # Generate and save test set
        test = Haldan_anis(L=self.L, bond=self.bond, keep = self.keep).generate_test_set()
        file_path_test_DMRG = os.path.join(self.path(), f'test_set_DMRG_partial_{len(self.keep)}spins.pkl')
        with open(file_path_test_DMRG, "wb") as f:
            pickle.dump(test, f)
        self._test_dataset_path = file_path_test_DMRG
        print(f"Test_set is generated in {time.time() - start_time:.2f} seconds.")

        return points, test

    def _load_dataset(self):
        
        path = self.path()

        train_path = os.path.join(path, f'train_set_DMRG_partial_{len(self.keep)}spins.pkl')
        test_path = os.path.join(path, f'test_set_DMRG_partial_{len(self.keep)}spins.pkl')

        if not os.path.exists(train_path):
            print('Ops! Train dataset not found. Generating it .....')
            train = Haldan_anis(L=self.L, bond=self.bond).points()
            with open(train_path, "wb") as f:
                pickle.dump(train, f)
        else:
            with open(train_path, 'rb') as f:
                train = pickle.load(f)
        self._train_dataset_path = train_path

        if not os.path.exists(test_path):
            print("Ops! Test dataset not found. Generating it .....")
            test = Haldan_anis(L=self.L, bond=self.bond).generate_test_set()
            with open(test_path, "wb") as f:
                pickle.dump(test, f)
        else:
            with open(test_path, 'rb') as f:
                test = pickle.load(f)       
        self._test_dataset_path = test_path

        return train, test

    def partial_density_matrix(self):
        print("Computing partial density matrices started .....")
        start_time = time.time()


        path = f"./dataset_L=51_bond=50_partial"

        file_path_train_DMRG = os.path.join(path, f'train_set_DMRG.pkl')
        with open(file_path_train_DMRG, "rb") as f:
            loaded_dataset = pickle.load(f)

        file_path_test_DMRG = os.path.join(path, f'test_set_DMRG.pkl')
        with open(file_path_test_DMRG, "rb") as f:
            loaded_test_set = pickle.load(f)
        #loaded_test_set = self._load_dataset()[1]
        Xte = loaded_test_set
        d1 = len(Xte)

        #loaded_dataset = self._load_dataset()[0]
        Xtr = loaded_dataset[0]
        d2 = len(Xtr)

        print(f"Tracing over training set ({d2} items)...")
        partial_rhos_train = Parallel(n_jobs=-1)(
            delayed(lambda x: x.partial_trace_to_dense_canonical(where=self.keep))(Xtr[i])
            for i in tqdm(range(d2), desc="Tracing train set"))
        
        '''file_path_partial_rhos_train = os.path.join(self.path(), 'partial_rhos_train.pkl')
        with open(file_path_partial_rhos_train, "wb") as f:
            pickle.dump(partial_rhos_train, f)'''

        print(f"Tracing over test set ({d1} items)...")    
        partial_rhos_test = Parallel(n_jobs=-1)(
            delayed(lambda x: x.partial_trace_to_dense_canonical(where=self.keep))(Xte[i])
            for i in tqdm(range(d1), desc="Tracing test set"))        
        
        '''file_path_partial_rhos_test = os.path.join(self.path(), 'partial_rhos_test.pkl')
        with open(file_path_partial_rhos_test, "wb") as f:
            pickle.dump(partial_rhos_test, f)'''

        print(f"Partial density matrices computed in {time.time() - start_time:.2f} seconds.")

        return partial_rhos_train, partial_rhos_test

    '''def _load_partial_density_matrix(self):
        
        path = self.path()

        partial_train_path = os.path.join(path, 'partial_rhos_train.pkl')
        partial_test_path = os.path.join(path, 'partial_rhos_test.pkl')

        if not os.path.exists(partial_train_path):
            print('Partian train dataset not found. It is generating .....')
            partial_train = self.partial_density_matrix()[0]
            with open(partial_train_path, "wb") as f:
                pickle.dump(partial_train, f)
        else:
            with open(partial_train_path, 'rb') as f:
                partial_train = pickle.load(f)
        self._partial_train_dataset_path = partial_train_path

        if not os.path.exists(partial_test_path):
            print("Partial test dataset not found. It is generating .....")
            partial_test = self.generate_test_set()[1]
            with open(partial_test_path, "wb") as f:
                pickle.dump(partial_test, f)
        else:
            with open(partial_test_path, 'rb') as f:
                partial_test = pickle.load(f)       
        self._partial_test_dataset_path = partial_test_path

        return partial_train, partial_test'''

    def gram_train_partial(self,partial_train):

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
        file_path_kernel_train_DMRG = os.path.join(self.path(), f"kernel_train_Haldane_DMRG_partial_from_{self.keep[0]}_to_{self.keep[-1]}_spins.hdf5")
        with h5py.File(file_path_kernel_train_DMRG, "w") as f:
            f.create_dataset("gram_train_DMRG_partial", data=gram)

        print(f"Gram matrix for training set computed in {time.time() - start_time:.2f} seconds.")
        return gram

    def gram_test_partial(self, partial_train, partial_test):

        print("Computing Gram matrix for testing set...")
        start_time = time.time()


        partial_rhos_train = partial_train
        partial_rhos_test = partial_test    

        '''partial_rhos_train = self.partial_density_matrix()[0]
        partial_rhos_test = self.partial_density_matrix()[1]

        partial_rhos_train = self._load_partial_density_matrix()[0]
        partial_rhos_test = self._load_partial_density_matrix()[1]'''
        d1 = len(partial_rhos_test)
        d2 = len(partial_rhos_train)
        gram_matrix_test = np.zeros((d1,d2))
        for i in tqdm(range(d1), desc='Gram Partial Test'):
            for j in range(d2):
                gram_matrix_test[i,j] = (np.trace(partial_rhos_test[i] @ partial_rhos_train[j]).real)**2
        file_path_kernel_test_DMRG = os.path.join(self.path(), f"kernel_test_Haldane_DMRG_partial_from_{self.keep[0]}_to_{self.keep[-1]}_spins.hdf5")
        with h5py.File(file_path_kernel_test_DMRG, "w") as f:
            f.create_dataset("gram_test_DMRG_partial", data = gram_matrix_test)
        print(f"Gram matrix for testing set computed in {time.time() - start_time:.2f} seconds.")
        return gram_matrix_test   
