import quimb as qu
import quimb.tensor as qtn
import numpy as np

class ANNNI:
    
    #---------------------------------------------------------------------------------#
    def __init__(self, L, ls):        
        self.L = L # number of particle
        self.ls = ls # the scale of dividing the range of h and k
    #---------------------------------------------------------------------------------#
    def MPO(self, h1, h2):
        I = qu.eye(2).real
        Z = qu.pauli('Z').real
        X = qu.pauli('X').real
        # define the MPO tensor
        W = np.zeros([5, 5, 2, 2], dtype = float)
        # allocate different values to each of the sites
        W[0, 0, :, :] = I
        W[0, 1, :, :] = X
        W[0, 2, :, :] = Z
        W[0, 4, :, :] = -h1 * X
        W[1, 4, :, :] = -h2 * X
        W[2, 3, :, :] = X
        W[3, 4, :, :] = -Z
        W[4, 4, :, :] = I
        Wl = W[0, :, :, :]
        Wr = W[:, 4, :, :]
        # build the MPO
        H = qtn.MatrixProductOperator([Wl] + [W] * (self.L - 2) + [Wr])
        return H
    #---------------------------------------------------------------------------------#
    def P(self):
        I = qu.eye(2).real
        Z = qu.pauli('Z').real

        # make projection
        W = np.zeros([2, 2, 2, 2], dtype = float)
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

        # make identity
        Identity = np.zeros([2, 2, 2, 2], dtype = int)
        Identity[0, 0, :, :] = I
        Identity[1, 1, :, :] = I
        Identity_side = np.zeros([2, 2, 2], dtype = int)
        Identity_side[0, :, :] = I
        Identity_side[1, :, :] = I
        Identity_side_minus = np.zeros([2, 2, 2], dtype = int)
        Identity_side_minus[0, :, :] = I
        Identity_side_minus[1, :, :] = -I
    

        # build projection odd and even
        
        #if int(self.L) % 2 != 0:

        even_form = [Identity]+[W]
        even_repeat = sum([even_form for i in range(int((self.L-3)/2))],[])

        odd_form = [W]+[Identity]
        odd_repeat = sum([odd_form for i in range(int((self.L-3)/2))],[])

        P_plus_even = qtn.MatrixProductOperator([Wlplus] +  even_repeat + [Identity] + [Wrplus])
        P_plus_odd = qtn.MatrixProductOperator([Identity_side] + odd_repeat + [W] + [Identity_side])
        P_minus_even = qtn.MatrixProductOperator([Wlminus] + even_repeat+ [Identity] + [Wrminus])
        P_minus_odd = qtn.MatrixProductOperator([Identity_side_minus] + odd_repeat + [W] + [Identity_side])

        #elif int(self.L) % 2 == 0:

            #even_form = [Identity]+[W]
            #even_repeat = sum([even_form for i in range(int((self.L-2)/2))],[])

            #odd_form = [W]+[Identity]
            #odd_repeat = sum([odd_form for i in range(int((self.L-2)/2))],[])

            #P_plus_even = qtn.MatrixProductOperator([Wlplus] +  even_repeat + [Identity_side])
            #P_plus_odd = qtn.MatrixProductOperator([Identity_side] + odd_repeat +  [Wrplus])
            #P_minus_even = qtn.MatrixProductOperator([Wlminus] + even_repeat+  [Identity_side] )
            #P_minus_odd = qtn.MatrixProductOperator([Identity_side_minus] + odd_repeat + [Wrminus])

        # build projection
        P_plus = qtn.MatrixProductOperator([Wlplus] + [W] * (self.L - 2) + [Wrplus])
        P_minus = qtn.MatrixProductOperator([Wlminus] + [W] * (self.L - 2) + [Wrminus])

        return P_plus_even, P_plus_odd, P_minus_even, P_minus_odd, P_plus, P_minus
    #---------------------------------------------------------------------------------#
    def DMRG(self, h1, h2):
        DMRG = qtn.tensor_dmrg.DMRG(ham = ANNNI(L = self.L, ls = self.ls).MPO(h1 = h1, h2 = h2), bond_dims = 150) 
        DMRG.solve(tol = 1e-3, verbosity = 0);
        ground_state = DMRG.state
        energy = DMRG.energy
        return ground_state, energy
    #---------------------------------------------------------------------------------#
    def generate_train_set(self):
        # Generate the dataset for the specified constant h1
        h1 = 1
        h2_value = np.linspace(-1.6, 1.6, int(self.ls))
        lst = []
        for h2v in h2_value:
            lst.append(h1)
            lst.append(h2v)

        # Generate final dataset which is based on (k, h) format
        result = np.array(lst).reshape(int(len(lst)/2), 2)

        # make projections
        P_plus_even, P_plus_odd, P_minus_even, P_minus_odd, P_plus, P_minus = ANNNI(L = self.L, ls = self.ls).P()
        
        lst_contract = []
        lst_target_projection = []

        #lst_contract_1 = []
        #lst_target_projection_1 = []

        lst_h1h2 = []
        lst_DMRG_state = []
        lst_target_DMRG = []

        for element in result:
            h1 = element[0]
            h2 = element[1]

            # the target part
            if h2 < -1.15:
                y = -1 # Antiferromagnetic
            elif h2 > 0:
                y = 1 # Paramagnetic   
            else:
                y = 0 # SPT
          		
            lst_h1h2.append(h1)
            lst_h1h2.append(h2)

            # the feature part
            DMRG_state, DMRG_energy = ANNNI(L = self.L, ls = self.ls).DMRG(h1 = h1, h2 = h2); # make DMRG state for these specific value of h and k
            lst_DMRG_state.append(DMRG_state) # DMRG states
            lst_target_DMRG.append(y)

            contraction_state_plus_even = P_plus_even.apply(DMRG_state); # projection state after P plus even projection
            normalisation_factor_plus_even = np.sqrt(np.abs(contraction_state_plus_even.H @ contraction_state_plus_even))
            if normalisation_factor_plus_even > 0.01:
                contraction_state_plus_even = contraction_state_plus_even/normalisation_factor_plus_even
                lst_contract.append(contraction_state_plus_even)
                lst_target_projection.append(y)

            contraction_state_plus_odd = P_plus_odd.apply(DMRG_state); # projection state after P plus odd projection
            normalisation_factor_plus_odd = np.sqrt(np.abs(contraction_state_plus_odd.H @ contraction_state_plus_odd))
            if normalisation_factor_plus_odd > 0.01:
                contraction_state_plus_odd = contraction_state_plus_odd/normalisation_factor_plus_odd
                lst_contract.append(contraction_state_plus_odd)
                lst_target_projection.append(y)

            contraction_state_minus_even = P_minus_even.apply(DMRG_state); # projection state after P minus even projection
            normalisation_factor_minus_even = np.sqrt(np.abs(contraction_state_minus_even.H @ contraction_state_minus_even))
            if normalisation_factor_minus_even > 0.01:
                contraction_state_minus_even = contraction_state_minus_even/normalisation_factor_minus_even
                lst_contract.append(contraction_state_minus_even)
                lst_target_projection.append(y)

            contraction_state_minus_odd = P_minus_odd.apply(DMRG_state); # projection state after P minus odd projection
            normalisation_factor_minus_odd = np.sqrt(np.abs(contraction_state_minus_odd.H @ contraction_state_minus_odd))
            if normalisation_factor_minus_odd > 0.01:
                contraction_state_minus_odd = contraction_state_minus_odd/normalisation_factor_minus_odd
                lst_contract.append(contraction_state_minus_odd)
                lst_target_projection.append(y)

            #contraction_state_plus = P_plus.apply(DMRG_state); # projection state after P minus projection
            #normalisation_factor_plus = np.sqrt(np.abs(contraction_state_plus.H @ contraction_state_plus))
            #if normalisation_factor_plus > 0.01:
            #    contraction_state_plus = contraction_state_plus/normalisation_factor_plus
            #    lst_contract_1.append(contraction_state_plus)
            #    lst_target_projection_1.append(y)

            #contraction_state_minus = P_minus.apply(DMRG_state); # projection state after P minus projection
            #normalisation_factor_minus = np.sqrt(np.abs(contraction_state_minus.H @ contraction_state_minus))
            #if normalisation_factor_minus > 0.01:
            #    lst_target_projection_1.append(y)

        DMRG_state = np.array(lst_DMRG_state)
        DMRG_target = np.array(lst_target_DMRG)
        project_state = np.array(lst_contract)
        projection_target = np.array(lst_target_projection)
        #project_state_1 = np.array(lst_contract_1)
        #projection_target_1 = np.array(lst_target_projection_1)
        h1h2 = np.array(lst_h1h2)

        return DMRG_state, DMRG_target, project_state, projection_target, h1h2
    

    def generate_test_set(self):
        # Generate the dataset for the specified constant h1
        h1_value = np.linspace((1e-5), 1.6, int(self.ls))
        h2_value = np.linspace(-1.6, 1.6, int(self.ls))
        lst = []
        for h1v in h1_value:
            for h2v in h2_value:
                lst.append(h1v)
                lst.append(h2v)

        # Generate final dataset which is based on (k, h) format
        result = np.array(lst).reshape(int(len(lst)/2), 2)

        lst_h1h2 = []
        lst_DMRG_state = []
        for element in result:
            h1 = element[0]
            h2 = element[1]

            lst_h1h2.append(h1)
            lst_h1h2.append(h2)

             # the feature part
            DMRG_state, DMRG_energy = ANNNI(L = self.L, ls = self.ls).DMRG(h1 = h1, h2 = h2); # make DMRG state for these specific value of h and k
            lst_DMRG_state.append(DMRG_state) # DMRG states


        DMRG_state = np.array(lst_DMRG_state)
        h1h2 = np.array(lst_h1h2)

        return DMRG_state, h1h2
