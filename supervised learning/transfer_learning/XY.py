import quimb as qu
import quimb.tensor as qtn
import numpy as np

class XY:
    
    #---------------------------------------------------------------------------------#
    def __init__(self, L, ls, j_value):        
        self.L = L # number of particle
        self.ls = ls # the scale of dividing the range of h and k
        self.j_value = -0.5
    #---------------------------------------------------------------------------------#
    def MPO(self, gamma, bz_value):
        H = qtn.MPO_ham_XY(self.L, j=(self.j_value*(1+gamma), self.j_value*(1-gamma)), bz=bz_value)
        return H
    #---------------------------------------------------------------------------------#
    def P(self):
        I = qu.eye(2).real
        Z = qu.pauli('Z').real
        # define the MPO tensor
        W = np.zeros([2, 2, 2, 2], dtype = float)
        # allocate different values for each site
        W[0, 0, :, :] = I
        W[1, 1, :, :] = Z
        Wr = np.zeros([2, 2, 2], dtype = float)
        Wr[0, :, :] = I
        Wr[1, :, :] = Z
        Wl = np.zeros([2, 2, 2], dtype = float)
        Wl[0, :, :] = I
        Wl[1, :, :] = -Z
        Wrplus = Wr
        Wlplus = Wr
        Wrminus = Wr
        Wlminus = Wl
        # build projection
        P_plus = qtn.MatrixProductOperator([Wlplus] + [W] * (self.L - 2) + [Wrplus])
        P_minus = qtn.MatrixProductOperator([Wlminus] + [W] * (self.L - 2) + [Wrminus])
        return P_plus, P_minus
    #---------------------------------------------------------------------------------#
    def DMRG(self, gamma, bz):
        DMRG = qtn.tensor_dmrg.DMRG(ham = XY(L = self.L, ls = self.ls).MPO(gamma, bz_value = bz), bond_dims = 20, cutoffs=1e-12) 
        DMRG.solve(tol = 1e-6, verbosity = 0);
        ground_state = DMRG.state
        energy = DMRG.energy
        return ground_state, energy
    #---------------------------------------------------------------------------------#
    def generate_dataset(self, j=None, bz=None):

        gamma_value = np.linspace((1e-5), 1, int(self.ls))
        bz_value = np.linspace((1e-5), 1, int(self.ls))
        lst = []
        for gamma in gamma_value:
            for bz in bz_value:
                lst.append(gamma)
                lst.append(bz)

        result = np.array(lst).reshape(int(len(lst)/2), 2)

        # make projections
        P_plus, P_minus  = XY(L = self.L, ls = self.ls).P()

        
        # making the feature part of the data
        lst_x = []
        lst_en =[]
        lst_contract =[]

        for element in result:
            gamma = element[0]
            bz = element[1]

            # the feature part
            DMRG_state, DMRG_energy = XY(L = self.L, ls = self.ls).DMRG(gamma = gamma, bz = bz)
            lst_x.append(DMRG_state) # DMRG states
            lst_en.append(DMRG_energy)# DMRG energy

            contraction_state_plus = P_plus.apply(DMRG_state); # projection state after P plus projection
            normalisation_factor_plus = np.sqrt(np.abs(contraction_state_plus.H @ contraction_state_plus))
            contraction_state_plus = contraction_state_plus/normalisation_factor_plus
            lst_contract.append(contraction_state_plus)


            #contraction_state_minus = P_minus.apply(DMRG_state); # projection state after P minus projection
            #normalisation_factor_minus = np.sqrt(np.abs(contraction_state_minus.H @ contraction_state_minus))
            #if normalisation_factor_minus > 0.01:
            #    contraction_state_minus = contraction_state_minus/normalisation_factor_minus
            #    lst_contract.append(contraction_state_minus)
            #    lst_y.append(y)
            #    lst_kh.append(k)
            #    lst_kh.append(h)

        X = np.array(lst_x)
        energy = np.array(lst_en)
        contract = np.array(lst_contract)

        return X, contract, energy
        
    
