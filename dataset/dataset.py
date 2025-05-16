import pandas as pd
import numpy as np
from matplotlib.path import Path 

class dataset:
    '''
    This class is used to load the dataset and process it.
    The path to the dataset is './original_phase_diagram.csv' 
    This .py file is tuned just by the original_phase_diagram.csv file
    please contact author if you do not have access to this file.
    CAUTION: DON'T MODIFY THE original_phase_diagram.csv.
    '''
    def __init__(self, data_path):
        self.data_path = './original_phase_diagram.csv' 
        self.data = pd.read_csv(data_path)
        self.points = self.data.drop_duplicates()
        self.points = self.points.dropna()
        self.points = self.points.reset_index(drop=True)

    def large_ex(self):
        '''
        This function is used to get the large_ex boundary data
        '''
        mask_1 = (self.points['y'] >= 0.48) & (self.points['y'] <= 2) & (self.points['x'] >= -2) & (self.points['x'] <= 0.4)
        first_part = self.points[mask_1].copy()
        first_part.sort_values(by='x', ascending=True, inplace=True)
        Large_Ex = pd.concat([first_part.loc[:85], first_part.loc[[120, 123]], first_part.loc[88:]])
        fix_row_1 = pd.DataFrame([[-2, 2]], columns=Large_Ex.columns)
        Large_Ex = pd.concat([Large_Ex, fix_row_1], ignore_index=True)
        Large_Ex.drop_duplicates(inplace=True)
        return Large_Ex

    def z_neel(self):
        '''
        This function is used to get the z_neel boundary data
        '''
        drop_idx_2 = [86,88]
        for i, point in self.points.iterrows():
            if point['x']>-0.3 or point['y']>0.825081 or point['y']<-0.825081:
                drop_idx_2.append(i)
        second_part = self.points.drop(drop_idx_2)
        zNeel = second_part.sort_values('y')
        return zNeel

    def large_ey(self):
        '''
        This function is used to get the large_ey boundary data
        '''
        mask_3 = (self.points['y'] <= -0.48) & (self.points['y'] >= -2) & (self.points['x'] >= -2.1) & (self.points['x'] <= 0.4)
        third_part = self.points[mask_3].copy()
        third_part.sort_values(by='x', ascending=True, inplace=True)
        Large_Ey = pd.concat([third_part.loc[:89], third_part.loc[[122, 121]], third_part.loc[86:]])
        fix_row_3 = pd.DataFrame([[-2.1, -2.1]], columns=Large_Ey.columns)
        Large_Ey = pd.concat([Large_Ey, fix_row_3], ignore_index=True)
        Large_Ey.drop_duplicates(inplace=True)
        return Large_Ey
    
    def y_neel(self):
        '''
        This function is used to get the y_neel boundary data'''
        mask_4_1 = (self.points['x'] >= -0.3) & (self.points['x'] <= 2.0) & (self.points['y'] >= 0.0) & (self.points['y'] <= 0.6)
        yNeel_1 = self.points[mask_4_1].copy()
        yNeel_1.sort_values(by='x', ascending=True, inplace=True)
        mask_4_2 = (self.points['x'] >= -0.7) & (self.points['x'] <= 0.5) & (self.points['y'] <= 2.0) & (self.points['y'] >= 0.30)
        yNeel_2 = self.points[mask_4_2].copy()
        yNeel_2.sort_values(by='y', ascending=False, inplace=True)
        fix_row_4 = pd.DataFrame([[2, 2]], columns=yNeel_1.columns)
        yNeel = pd.concat([yNeel_2, yNeel_1, fix_row_4], ignore_index=True)
        yNeel.drop_duplicates(inplace=True)
        return yNeel
    
    def large_d(self):
        '''
        This function is used to get the large_d boundary data
        '''
        mask_5 = (self.points['x'] >= 0.9) & (self.points['x'] <= 2.0) & (self.points['y'] >= -1) & (self.points['y'] <= 1)
        Large_D = self.points[mask_5].copy()
        Large_D.sort_values(by='y', ascending = True, inplace = True)
        Large_D.drop_duplicates(inplace=True)
        return Large_D

    def x_neel(self):
        '''
        This function is used to get the x_neel boundary data
        '''
        mask_6_1 = (self.points['x'] >= -0.3) & (self.points['x'] <= 2.0) & (self.points['y'] >= -0.6) & (self.points['y'] <= 0.0)
        xNeel_1 = self.points[mask_6_1].copy()
        xNeel_1.sort_values(by='x', ascending = False, inplace = True)
        mask_6_2 = (self.points['x'] >= -0.7) & (self.points['x'] <= 0.5) & (self.points['y'] <= -0.3) & (self.points['y'] >= -2.1)
        xNeel_2 = self.points[mask_6_2].copy()
        xNeel_2.sort_values(by='y', ascending = False, inplace = True)
        fix_row_6 = pd.DataFrame([[2, -2.1]], columns=xNeel_1.columns)
        xNeel = pd.concat([xNeel_1,xNeel_2, fix_row_6], ignore_index=True)
        xNeel.drop_duplicates(inplace=True)
        return xNeel

    def scatter_points(self, r = 0.01):
        '''
        This function is used to get the scatter points data include their 
        exact x and y coordinates and their corresponding labels which are 
        0,1,2,3,4,5,6 for Haldane, Large_Ex, zNeel, Large_Ey, yNeel, Large_D 
        and xNeel respectively. Here by default r = 0.01 which represent the 
        line space between -2 to 2 with step size of 0.01 among x and y axis.
        users can change the value of r to get the desired step size.
        '''
        polygon1 = Path(self.large_ex())
        polygon2 = Path(self.z_neel())
        polygon3 = Path(self.large_ey())
        polygon4 = Path(self.y_neel())
        polygon5 = Path(self.large_d())
        polygon6 = Path(self.x_neel())

        x = np.arange(-2, 2, r)
        y = np.arange(-2, 2, r)
        X, Y = np.meshgrid(x, y)
        points = np.vstack((X.ravel(), Y.ravel())).T

        zone1 = polygon1.contains_points(points)  
        zone2 = polygon2.contains_points(points)  
        zone3 = polygon3.contains_points(points)  
        zone4 = polygon4.contains_points(points)  
        zone5 = polygon5.contains_points(points)  
        zone6 = polygon6.contains_points(points) 

        labels = np.zeros(points.shape[0], dtype=int)

        labels[zone1] = 1
        labels[zone2] = 2
        labels[zone3] = 3
        labels[zone4] = 4
        labels[zone5] = 5
        labels[zone6] = 6

        labels = labels.reshape(X.shape)

        return labels, X, Y

    def get_boundary(self):
        '''
        This function is used to get the boundary data of all the 
        boundaries in the dataset. The boundaries are Large_Ex, zNeel, 
        Large_Ey, yNeel, Large_D and xNeel. The function returns a 
        the coordinate of three main lines in Anisotropic Haldane Chain.
        '''
        drop_idx_1 = [86,88]
        for i, point in self.points.iterrows():
            if point[0]>-0.3:
                drop_idx_1.append(i)
            elif point[1]>0.825081:
                drop_idx_1.append(i)
            elif point[1]<-0.825081:
                drop_idx_1.append(i)
        first_part = self.points.drop(drop_idx_1)
        first_part = first_part.sort_values('y')
        lst_1 = first_part.index.to_list()

        drop_idx_2 = []
        drop_idx_2.extend(lst_1)
        for i, point in self.points.iterrows():
            if point[0]>1 or point[0]<-1:
                drop_idx_2.append(i)
        drop_idx_2.remove(120)
        drop_idx_2.remove(122)
        second_part = self.points.drop(drop_idx_2)
        second_part = second_part.sort_values('y')
        lst_2 = second_part.index.to_list()

        drop_idx_3 =[]
        for i, point in self.points.iterrows():
            if point[0]<0.7:
                drop_idx_3.append(i)
        third_part = self.points.drop(drop_idx_3)
        third_part = third_part.sort_values('y')

        return first_part, second_part, third_part