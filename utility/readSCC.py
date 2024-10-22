import xml.etree.ElementTree as ET
# from . import coordinate
from typing import List
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('_mpl-gallery')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from scipy import integrate

from utility import coordinate
from utility import geometry
from utility import stratumModel

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)

from MCNPInput import MCNPAutoInput, GeometricModel, Material

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

# 为CIFLog特化的轨迹和模型类
class CIFLogTrack(stratumModel.WellTrack):
    def __init__(self, hd_Path:str, vd_Path:str):
        '''
        从对应的文件读取井眼轨迹
        :param hd_Path:
        :param vd_Path:
        '''
        super().__init__()
        hd = np.loadtxt(hd_Path,float, skiprows=2)
        vd = np.loadtxt(vd_Path, float, skiprows=2)
        self.startDepth = hd.shape[0] - 1
        for i in range(hd.shape[0]):
            if hd[i, 1] > 0:
                self.startDepth = i - 1
                break
        track = np.array([vd[:,0], vd[:, 1], hd[:, 1]]).T * 100 # 程序默认单位为cm，ciflog默认单位是m
        self.trackPD = pd.DataFrame(track, columns=["Depth", "Y", "X"]).set_index("Depth")
        Inclination_I = reverse_inclination(self.trackPD.iloc[self.startDepth:])
        Inclination_array = np.zeros((self.trackPD.shape[0]))
        Inclination_array[self.startDepth:] = Inclination_I
        self.trackPD["Inc"] = Inclination_array
        self.trackPD["Azi"] = np.zeros_like(Inclination_array)
        self.trackPD["Z"] = np.zeros_like(Inclination_array)
        self.trackPD = self.trackPD[["X", "Y", "Z", "Inc", "Azi"]]

class LayerModel(stratumModel.StratumModel_DataFrame):
    def __init__(self, file:str, trajectory: stratumModel.WellTrack, radius_of_investigation:float = 80, well_diameter:float = 10):
        _CIFLogSCC = ET.parse(file)
        _layerLinesTree = _CIFLogSCC.find("LayerLines")
        _controlLinesTree = _CIFLogSCC.find("ControlLines")
        api: np.ndarray = np.empty((0,))
        colHead = []
        depth_y = []
        _x_list = []
        aptList = []
        _layerLines = _layerLinesTree.findall("LayerLine")
        for layerLine in _layerLines:
            _pro_str = layerLine.attrib["PropertyValues"]
            _por_ListStr = _pro_str.split(",")
            _por_ListFloat = list(map(float, _por_ListStr))
            aptList.append(_por_ListFloat[0])
            colHead.append(layerLine.attrib["ID"])
        api = np.array(aptList)
        api_na = np.isclose(api, -99999.)
        api[api_na] = np.nan
        for controlLine in _controlLinesTree:
            x = float(controlLine.attrib["X"])
            _x_list.append(x)
            _y_list = []
            for pointData in controlLine.findall("PointData"):
                y = float(pointData.attrib["y"])
                _y_list.append(y)
            depth_y.append(_y_list)
        depth_array = np.array(depth_y)*100
        x_idx = np.array(_x_list)*100
        data = pd.DataFrame(depth_array, index=x_idx)
        super().__init__(data,api, trajectory, radius_of_investigation, well_diameter)


