import quimb as qu
import quimb.tensor as qtn
import numpy as np
from joblib import Parallel, delayed


class Haldan_anis:

    #-----------------------------------------------------------------------#
    def __init__(self, L):
        self.L = L 
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
    def P(self):
        I = qu.eye(3).real
        Z = qu.pauli('Z',dim=3).real

        W = np.zeros([2, 2, 3, 3], dtype = float)
        W[0, 0, :, :] = I
        W[1, 1, :, :] = Z

        Wr = np.zeros([2, 3, 3], dtype = float)
        Wr[0, :, :] = I
        Wr[1, :, :] = Z

        Wl = np.zeros([2, 3, 3], dtype = float)
        Wl[0, :, :] = I
        Wl[1, :, :] = -Z

        Wrplus = Wr
        Wlplus = Wr
        Wrminus = Wr
        Wlminus = Wl    

        Identity = np.zeros([2, 2, 3, 3], dtype = int)
        Identity[0, 0, :, :] = I
        Identity[1, 1, :, :] = I

        Identity_side = np.zeros([2, 3, 3], dtype = int)
        Identity_side[0, :, :] = I
        Identity_side[1, :, :] = I
        
        Identity_side_minus = np.zeros([2, 3, 3], dtype = int)
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
    #-----------------------------------------------------------------------#
    def DMRG(self, d1, e1):
        dmrg_solver = qtn.tensor_dmrg.DMRG(ham = self.MPO(D = d1, E = e1), bond_dims = 80) 
        dmrg_solver.solve(tol = 1e-3, verbosity = 0);
        ground_state = dmrg_solver.state
        return ground_state
    #-----------------------------------------------------------------------#
    def points(self):
        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)
        lst_points = []
        lst_target = []

        for e in E:
            if 0.8 < e < 2:
                lst_points.append([-2, e])
                lst_target.append(1)  # 'large_ex'
            elif -0.8 < e < 0.8:
                lst_points.append([-2, e])
                lst_target.append(3)  # 'z_neel'
            elif -2 < e < -0.8:
                lst_points.append([-2, e])
                lst_target.append(2)  # 'large_ey'

        for e in E:
            if -2 < e < -0.4:
                lst_points.append([2, e])
                lst_target.append(4)  # 'x_neel'
            elif -0.4 < e < 0.4:
                lst_points.append([2, e])
                lst_target.append(5)  # 'large_d'
            elif 0.4 < e < 2.0:
                lst_points.append([2.0, e])
                lst_target.append(6)  # 'y_neel'
                
        for d in D:
            if -2 < d < 0.2:
                lst_points.append([d, 2])
                lst_target.append(1)  # 'large_ex'
            elif 0.2 < d < 2.0:
                lst_points.append([d, 2.0])
                lst_target.append(6)  # 'y_neel'

        for d in D:
            if -2 < d < 0.2:
                lst_points.append([d, -2])
                lst_target.append(2)  # 'large_ey'
            elif 0.2 < d < 2.0:
                lst_points.append([d, -2])
                lst_target.append(4)  # 'x_neel'

        for d in np.arange(-2, -0.5, 0.1):  
            lst_points.append([d, 0.0]) # 'z_neel'
            lst_target.append(3) 

        for d in np.arange(0.9, 2, 0.1):
            lst_points.append([d, 0.0])
            lst_target.append(5)  # 'large_d'

        for d in np.arange(-0.2, 0.6, 0.1):
            lst_points.append([d, 0.0])
            lst_target.append(7) #'Haldane'


        points = np.array(lst_points)
        targets = np.array(lst_target)
        def compute_dmrg(d, e):
            return self.DMRG(d1=d, e1=e)

        lst_DMRG = Parallel(n_jobs=5, backend = 'threading')(delayed(compute_dmrg)(point[0], point[1]) for point in points)

        return lst_DMRG, targets

    def apply_projection(self):
        P_plus_even, P_plus_odd, P_minus_even, P_minus_odd, _, _ = self.P()
        state, target = self.points()

        lst_target = []
        lst_state = []
        for proj in [P_plus_even, P_plus_odd, P_minus_even, P_minus_odd]:
            for s, t in zip(state, target):
                lst_state.append(proj.apply(s))
                lst_target.append(t)

        return lst_state, lst_target


    '''def generate_train_set(self, n_jobs=5):
        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)
        
        projections = Haldan_anis(L=self.L, ls=self.ls).P()[:4]  # Extract relevant projection operators
        
        






        # Parallel execution over E
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_point)(d1, e, target, projections)
            for condition, d1, target in conditions
            for e in E if condition(e)
        )
        
        # Parallel execution over D
        results += Parallel(n_jobs=n_jobs)(
            delayed(self.process_point)(d, e1, target, projections)
            for condition, d, target in d_conditions
            for e1 in [2, -2, 0.0] if condition(d)
        )
        
        # Unpack results
        lst_points, lst_DMRG, lst_target, lst_contract = [], [], [], []
        
        for points, state, target, projections in results:
            lst_points.append(points)
            lst_DMRG.append(state)
            lst_target.append(target)
            for proj_state, proj_target in projections:
                lst_contract.append((proj_state, proj_target))
        
        return (
            np.array(lst_DMRG),
            np.array(lst_target),
            np.array([p[0] for p in lst_contract]),
            np.array([p[1] for p in lst_contract]),
            np.array(lst_points),
        )'''

    
    def generate_test_set(self):
        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)
        
        def compute_dmrg(d, e):
            return self.DMRG(d1=d, e1=e)

        results = Parallel(n_jobs=5, backend = 'loky')(delayed(compute_dmrg)(d, e) for e in E for d in D)
        
        return results
