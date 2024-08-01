import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable
import scipy.optimize as opt
import scienceplots
import latex
plt.style.use('science')

#最小曲率法2维
def minimum_curvature_method(dm:float, I_1:float, I_2:float):
    '''

    :param dm: measure depth
    :param I_1: inclination of first survey
    :param I_2: inclination of second survey
    :return: vertical displacement, horizontal displacement
    '''
    DL = I_2 - I_1
    if DL == 0.:
        return np.cos(I_1) * dm, np.sin(I_1) * dm
    R = dm / DL
    d_Vert = (np.sin(I_2) - np.sin(I_1)) * R
    d_Hori = (np.cos(I_1) - np.cos(I_2)) * R
    return d_Vert, d_Hori


#直接连接的方法求解井斜
def direct_connection_solution(Vert:float, Hori:float):
    '''

    :param Vert: vertical displacement
    :param Hori: Horizontal displacement
    :return: inclination of first survey
    '''
    if Hori == 0:
        return 0.
    if Vert == 0.:
        return np.pi/2
    r = np.arctan(Hori / Vert)
    if r < 0:
        r = r + np.pi
    return r

#求解反函数方程求井斜
def inverse_function_solution(I_1:float, DM:float, Vert:float, Hori:float):
    '''

    :param I_1: inclination of first survey
    :param DM: measure depth
    :param Vert: vertical displacement
    :param Hori: horizontal displacement
    :return: inclination of second survey
    '''
    if Hori == 0. and I_1 == 0.:
        return 0.
    def V(X:float):
        if X == 0.:
            return DM - Vert
        return (np.sin(X) - np.sin(I_1)) * DM / (X - I_1) - Vert

    def H(X:float):
        if X == 0.:
            return 0.
        return (np.cos(X) - np.cos(I_1)) * DM / (X - I_1) - Hori

    x_1 = opt.root(V, 0)
    x_2 = opt.root(H, 0)

    uni_result = set(x_1.x) & set(x_2.x)

    if len(uni_result) >= 1:
        for i in uni_result:
            return i
    else:
        for i in x_1.x:
            for j in x_2.x:
                if abs(i - j) < 1e-2:
                    return (i+j) / 2
    return direct_connection_solution(Vert, Hori)

def reverse_inclination(data:pd.DataFrame):
    '''

    :param data: dataframe, index is measure depth, columns are [vertical coordinate, horizontal coordinate]
    :return: inclination as np.array
    '''
    result_Direct_I = np.empty((data.shape[0],))
    for i in range(data.shape[0]-1):
        D_V = data.iloc[i+1,0] - data.iloc[i, 0]
        D_H = data.iloc[i+1, 1] - data.iloc[i, 1]
        I_2 = direct_connection_solution(D_V, D_H)
        result_Direct_I[i] = I_2

    result_Direct_I[-1] = result_Direct_I[-2]
    return result_Direct_I



#绘图代码留档
def plot_two_methods_trajectory(data:pd.DataFrame):
    '''

    :param data: dataframe, index is measure depth, columns are [vertical coordinate, horizontal coordinate]
    :return:
    '''
    I_1 = 0.
    startV = data.iloc[0, 0]
    startH = data.iloc[0, 1]
    result_Direct_I = np.empty((data.shape[0]-1,))
    direct_point = np.empty(data.shape)
    invert_point = np.empty(data.shape)
    for i in range(data.shape[0]-1):
        MD = data.index[i+1] - data.index[i]
        D_V = data.iloc[i+1,0] - data.iloc[i, 0]
        D_H = data.iloc[i+1, 1] - data.iloc[i, 1]
        I_2 = inverse_function_solution(I_1, MD, D_V, D_H)
        d_v,d_h = minimum_curvature_method(MD, I_1, I_2)
        invert_point[i, 0] = startV
        startV += d_v
        invert_point[i, 1] = startH
        startH += d_h
        result_Direct_I[i] = I_2
        I_1 = I_2
        if i == 357:
            print("poont")
    I_1 = 0.
    startV = data.iloc[0, 0]
    startH = data.iloc[0, 1]
    for i in range(data.shape[0]-1):
        MD = data.index[i+1] - data.index[i]
        D_V = data.iloc[i+1,0] - data.iloc[i, 0]
        D_H = data.iloc[i+1, 1] - data.iloc[i, 1]
        I_2 = direct_connection_solution(D_V, D_H)
        d_v,d_h = minimum_curvature_method(MD, I_1, I_2)
        direct_point[i, 0] = startV
        startV += d_v
        direct_point[i, 1] = startH
        startH += d_h
        result_Direct_I[i] = I_2
        I_1 = I_2
        if i == 357:
            print("poont")

    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    # ax.title("direct solution")
    ax.plot(data.iloc[:-1,1], data.iloc[:-1, 0], label = "actual trajectory")
    ax.plot(direct_point[:-1, 1], direct_point[:-1, 0], label = "direct solution trajectory")
    ax.set(xlabel="horizontal displacement(m)")
    ax.set(ylabel="vertical displacement(m)")
    ax.autoscale(tight=True)
    ax.legend()
    ax.grid()
    ax.invert_yaxis()

    ax = axs[1]
    ax.plot(data.iloc[:-1, 1], data.iloc[:-1, 0], label="actual trajectory")
    ax.plot(invert_point[:-1, 1], invert_point[:-1, 0], label = "invert method trajectory ")
    # ax.plot(data.index.values[:400], direct_point[:400, 0] - data.iloc[:400, 0])
    # ax.plot(data.index.values[:400], direct_point[:400, 1] - data.iloc[:400, 1])
    ax.set(xlabel="horizontal displacement(m)")
    ax.set(ylabel="vertical displacement(m)")
    ax.autoscale(tight=True)
    ax.legend()
    ax.grid()
    ax.invert_yaxis()

    plt.show()

class Trackofwell():
    def __init__(self, hd_Path:str, vd_Path:str):
        hd = np.loadtxt(hd_Path,float, skiprows=2)
        vd = np.loadtxt(vd_Path, float, skiprows=2)
        self.startDepth = hd.shape[0] - 1
        for i in range(hd.shape[0]):
            if hd[i, 1] > 0:
                self.startDepth = i - 1
                break
        track = np.array([vd[:,0], vd[:, 1], hd[:, 1]]).T
        self._trackPD = pd.DataFrame(track,columns=["measureDepth", "VD", "HD"]).set_index("measureDepth")
        Inclination_I = reverse_inclination(self._trackPD.iloc[self.startDepth:])
        Inclination_array = np.zeros((self._trackPD.shape[0]))
        Inclination_array[self.startDepth:] = Inclination_I
        self._trackPD["Inclination"] = Inclination_array

    @property
    def track(self) -> pd.DataFrame:
        return self._trackPD.iloc[self.startDepth:]

if __name__ == '__main__':
    t = Trackofwell("../otherThings/HDRIFT.txt", "../otherThings/VDRIFT.txt")
    print(t.track)
