import xml.etree.ElementTree as ET
# from . import coordinate
from typing import List
import pandas as pd
from copy import deepcopy

from utility.coordinate import Point
from utility.readTrack import Trackofwell
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from scipy import integrate

from utility import coordinate
from utility import geometry
from utility import stratumModel

from MCNPInput import MCNPAutoInput, GeometricModel, Material

#小模型
class smallModel(object):
    def __init__(self, stratum_border: list[stratumModel.stratum], zone:stratumModel.DetectionZone):
        self.stratumBorder = stratum_border
        self.zone = zone

def twoDLineFunction(p1: coordinate.Point, p2:coordinate.Point):
    if p1[0] == p2[0]:
        def rvl(x):
            return p1[1]
    else:
        def rvl(x):
            return (x - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[0]) - p1[1]
    return rvl

class LayerModel(object):
    def __init__(self, file:str, trajectory: Trackofwell, radius_of_investigation:float = 80, well_diameter: float = 10):
        self._CIFLogSCC = ET.parse(file)
        self._layerLinesTree = self._CIFLogSCC.find("LayerLines")
        self._controlLinesTree = self._CIFLogSCC.find("ControlLines")
        self.data:pd.DataFrame = pd.DataFrame()
        self.api:np.ndarray = np.empty((0,))
        self._load()
        self.track = trajectory

        # 探测范围
        self.radius = radius_of_investigation
        # 井眼大小
        self.wellDiameter = well_diameter

    def _load(self):
        colHead = []
        depth_y = []
        _x_list = []
        aptList = []
        _layerLines = self._layerLinesTree.findall("LayerLine")
        for layerLine in _layerLines:
            _pro_str = layerLine.attrib["PropertyValues"]
            _por_ListStr = _pro_str.split(",")
            _por_ListFloat = list(map(float, _por_ListStr))
            aptList.append(_por_ListFloat[0])
            colHead.append(layerLine.attrib["ID"])
        self.api = np.array(aptList)
        for conctralLine in self._controlLinesTree:
            x = float(conctralLine.attrib["X"])
            _x_list.append(x)
            _y_list = []
            for pointData in conctralLine.findall("PointData"):
                y = float(pointData.attrib["y"])
                _y_list.append(y)
            depth_y.append(_y_list)
        depth_array = np.array(depth_y)

        self.data = pd.DataFrame(depth_array, index=_x_list)

    def __len__(self):
        return self.data.shape[1]

    def getSmallModel(self, index:int):
        if index < 0 or index >= self.track.track.shape[0]:
            return None
        # 探测点坐标
        p = self.track.track.iloc[index].values # vd, hd, inclination
        # 建立一个探测器相对坐标系
        coord = coordinate.CoordinateSystem().move_to(coordinate.ThreeDVector(p[1], p[0], 0.)).rotate(coordinate.ThreeDVector(0,0,1),coordinate.radian_to_angle(-p[2]))
        dZone = stratumModel.DetectionZone(coord, self.radius, self.wellDiameter)
        # 探测范围四个端点（在相对坐标系中建立，转换回原始坐标系）
        endPoint0 = coordinate.Point(self.radius, self.radius, 0, coord = coord).get_point_in_origin()
        endPoint1 = coordinate.Point(self.radius, -self.radius, 0, coord=coord).get_point_in_origin()
        endPoint2 = coordinate.Point(-self.radius, -self.radius, 0, coord=coord).get_point_in_origin()
        endPoint3 = coordinate.Point(-self.radius, self.radius, 0, coord= coord).get_point_in_origin()
        # 探测范围的x范围，y范围
        x_max = max(endPoint0[0], endPoint1[0], endPoint2[0], endPoint3[0])
        x_min = min(endPoint0[0], endPoint1[0], endPoint2[0], endPoint3[0])
        y_max = max(endPoint0[1], endPoint1[1], endPoint2[1], endPoint3[1])
        y_min = min(endPoint0[1], endPoint1[1], endPoint2[1], endPoint3[1])
        # 找到模型在哪两条控制线之间
        index_x_min = self.data.index.values.searchsorted(x_min) - 1
        index_x_max = self.data.index.values.searchsorted(x_max)
        # 如果探测范围中穿过了一条以上的控制线，则抛出地层界面变化过快的异常
        if index_x_max - index_x_min > 2:
            raise Exception("The stratum interface changes rapidly.")

        # 第一个地层上边界为None
        upBoundary = None

        stratumList = []
        for j in range(self.data.shape[1] - 1, 0, -1):
            # 这里第一版出现了一个问题就是CIFLog中给出的y是朝向地心的，最开始以错误的顺序访问了地层边界，现象改为从数据帧以逆向顺序可以避免大范围修改
            # 这个循环内是计算一个地层的，循环每一个地层
            # 下界面
            lowerBoundary = []
            for i in range(index_x_min, index_x_max):
                # 测量点在地层描述的水平边界外，认为描述模型外的所有地层水平
                if index_x_min < 0:
                    controlPoint_0 = coordinate.Point(p[1] - self.radius, self.data.iloc[0, j], 0)
                    controlPoint_1 = coordinate.Point(self.data.index.values[0], self.data.iloc[0, j], 0)
                    # 下界面方向向量应该y轴负方向，z应为正
                    dipDirectionPoint = coordinate.Point(self.data.index.values[0], self.data.iloc[0, j], 1)
                    plane = geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint)
                else:
                    controlPoint_0 = coordinate.Point(self.data.index.values[i], self.data.iloc[i, j], 0)
                    controlPoint_1 = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, j], 0)
                    # 下界面方向向量应该y轴负方向，z应为正
                    dipDirectionPoint = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, j], 1)
                    plane = geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint)
                lowerBoundary.append(plane)
            # TEST://
            #建立地层对象
            if len(lowerBoundary) == 1:
                lowB = stratumModel.OnePlaneBoundary(lowerBoundary[0], dZone)
            elif len(lowerBoundary) == 2:
                lowB_1 = stratumModel.OnePlaneBoundary(lowerBoundary[0], dZone)
                lowB_2 = stratumModel.OnePlaneBoundary(lowerBoundary[1], dZone)
                if np.dot(lowerBoundary[0].normal, lowerBoundary[1].normal) > 0:
                    lowB = stratumModel.UnionAndPlanesBoundary(lowB_1, lowB_2)
                else:
                    lowB = stratumModel.UnionOrPlanesBoundary(lowB_1, lowB_2)
            if upBoundary is not None:
                if len(upBoundary) == 1:
                    upB = stratumModel.OnePlaneBoundary(upBoundary[0], dZone)
                elif len(upBoundary) == 2:
                    upB_1 = stratumModel.OnePlaneBoundary(upBoundary[0], dZone)
                    upB_2 = stratumModel.OnePlaneBoundary(upBoundary[1], dZone)
                    if np.dot(upBoundary[0].normal, upBoundary[1].normal) > 0:
                        upB = stratumModel.UnionAndPlanesBoundary(upB_1, upB_2)
                    else:
                        upB = stratumModel.UnionOrPlanesBoundary(upB_1, upB_2)
                B = stratumModel.UnionAndPlanesBoundary(upB, lowB)
            else:
                B = lowB
            stratum = stratumModel.stratum(B, self.api[j - 1], dZone)
            if not np.isclose(stratum.volume, 0):
                stratumList.append(stratum)
            # 下界面将会变成下一个地层的上界面，需要将界面的法向量反转
            upBoundary = list(map(lambda x: -x, lowerBoundary))

        #如果 体积不等于零的地层 数量不为零
        if len(stratumList) != 0:
            return smallModel(stratumList, dZone)
        else:
            return None


class _LayerModel(object):
    def __init__(self, file:str, trajectory: Trackofwell, radius_of_investigation:float = 80, well_diameter: float = 10):
        self._CIFLogSCC = ET.parse(file)
        self._layerLinesTree = self._CIFLogSCC.find("LayerLines")
        self._controlLinesTree = self._CIFLogSCC.find("ControlLines")
        self._data:pd.DataFrame = pd.DataFrame()
        self._api:np.ndarray = np.empty((0,))
        self._load()
        self.track = trajectory

        # 探测范围
        self._radius = radius_of_investigation
        # 井眼大小
        self._wellDiameter = well_diameter

    def _load(self):
        colHead = []
        depth_y = []
        _x_list = []
        aptList = []
        _layerLines = self._layerLinesTree.findall("LayerLine")
        for layerLine in _layerLines:
            _pro_str = layerLine.attrib["PropertyValues"]
            _por_ListStr = _pro_str.split(",")
            _por_ListFloat = list(map(float, _por_ListStr))
            aptList.append(_por_ListFloat[0])
            colHead.append(layerLine.attrib["ID"])
        self._api = np.array(aptList)
        for conctralLine in self._controlLinesTree:
            x = float(conctralLine.attrib["X"])
            _x_list.append(x)
            _y_list = []
            for pointData in conctralLine.findall("PointData"):
                y = float(pointData.attrib["y"])
                _y_list.append(y)
            depth_y.append(_y_list)
        depth_array = np.array(depth_y)

        self._data = pd.DataFrame(depth_array, index=_x_list)

    @property
    def layerSize(self) -> int:
        return self._data.shape[1]

    def getSmallModel(self, index:int):
        if index < 0 or index >= self.track.track.shape[0]:
            return None
        # 探测点坐标
        p = self.track.track.iloc[index].values # vd, hd, inclination
        # 建立一个探测器相对坐标系
        coord = coordinate.CoordinateSystem().move_to(coordinate.ThreeDVector(p[1], p[0], 0.)).rotate(coordinate.ThreeDVector(0,0,1),coordinate.radian_to_angle(-p[2]))
        dZone = stratumModel.DetectionZone(coord, self._radius, self._wellDiameter)
        # 探测范围四个端点（在相对坐标系中建立，转换回原始坐标系）
        endPoint0 = coordinate.Point(self._radius, self._radius, 0, coord = coord).get_point_in_origin()
        endPoint1 = coordinate.Point(self._radius, -self._radius, 0, coord=coord).get_point_in_origin()
        endPoint2 = coordinate.Point(-self._radius, -self._radius, 0, coord=coord).get_point_in_origin()
        endPoint3 = coordinate.Point(-self._radius, self._radius, 0, coord= coord).get_point_in_origin()
        # 探测范围的x范围，y范围
        x_max = max(endPoint0[0], endPoint1[0], endPoint2[0], endPoint3[0])
        x_min = min(endPoint0[0], endPoint1[0], endPoint2[0], endPoint3[0])
        y_max = max(endPoint0[1], endPoint1[1], endPoint2[1], endPoint3[1])
        y_min = min(endPoint0[1], endPoint1[1], endPoint2[1], endPoint3[1])
        # 找到模型在哪两条控制线之间
        index_x_min = self._data.index.values.searchsorted(x_min)-1
        index_x_max = self._data.index.values.searchsorted(x_max)
        # 如果探测范围中穿过了一条以上的控制线，则抛出地层界面变化过快的异常
        if index_x_max - index_x_min > 2:
            raise Exception("The stratum interface changes rapidly.")

        # 第一个地层上边界为None
        upBoundary = None

        stratumList = []
        for j in range(self._data.shape[1]-1, 0, -1):
            # 这里第一版出现了一个问题就是CIFLog中给出的y是朝向地心的，最开始以错误的顺序访问了地层边界，现象改为从数据帧以逆向顺序可以避免大范围修改
            # 这个循环内是计算一个地层的，循环每一个地层
            # 下界面
            lowerBoundary = []
            for i in range(index_x_min, index_x_max):
                # 测量点在地层描述的水平边界外，认为描述模型外的所有地层水平
                if index_x_min < 0:
                    controlPoint_0 = coordinate.Point(p[1] - self._radius, self._data.iloc[0, j], 0)
                    controlPoint_1 = coordinate.Point(self._data.index.values[0], self._data.iloc[0, j], 0)
                    # 下界面方向向量应该y轴负方向，z应为正
                    dipDirectionPoint = coordinate.Point(self._data.index.values[0], self._data.iloc[0, j], 1)
                    plane = geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint)
                else:
                    controlPoint_0 = coordinate.Point(self._data.index.values[i], self._data.iloc[i, j], 0)
                    controlPoint_1 = coordinate.Point(self._data.index.values[i+1], self._data.iloc[i+1, j], 0)
                    # 下界面方向向量应该y轴负方向，z应为正
                    dipDirectionPoint = coordinate.Point(self._data.index.values[i+1], self._data.iloc[i+1, j], 1)
                    plane = geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint)
                lowerBoundary.append(plane)
            # TEST://
            #建立地层对象
            if len(lowerBoundary) == 1:
                lowB = stratumModel.OnePlaneBoundary(lowerBoundary[0], dZone)
            elif len(lowerBoundary) == 2:
                lowB_1 = stratumModel.OnePlaneBoundary(lowerBoundary[0], dZone)
                lowB_2 = stratumModel.OnePlaneBoundary(lowerBoundary[1], dZone)
                if np.dot(lowerBoundary[0].normal, lowerBoundary[1].normal) > 0:
                    lowB = stratumModel.UnionAndPlanesBoundary(lowB_1, lowB_2)
                else:
                    lowB = stratumModel.UnionOrPlanesBoundary(lowB_1, lowB_2)
            if upBoundary is not None:
                if len(upBoundary) == 1:
                    upB = stratumModel.OnePlaneBoundary(upBoundary[0], dZone)
                elif len(upBoundary) == 2:
                    upB_1 = stratumModel.OnePlaneBoundary(upBoundary[0], dZone)
                    upB_2 = stratumModel.OnePlaneBoundary(upBoundary[1], dZone)
                    if np.dot(upBoundary[0].normal, upBoundary[1].normal) > 0:
                        upB = stratumModel.UnionAndPlanesBoundary(upB_1, upB_2)
                    else:
                        upB = stratumModel.UnionOrPlanesBoundary(upB_1, upB_2)
                B = stratumModel.UnionAndPlanesBoundary(upB, lowB)
            else:
                B = lowB
            stratum = stratumModel.stratum(B, self._api[j-1], dZone)
            if not np.isclose(stratum.volume, 0):
                stratumList.append(stratum)
            # 下界面将会变成下一个地层的上界面，需要将界面的法向量反转
            upBoundary = list(map(lambda x: -x, lowerBoundary))

        #如果 体积不等于零的地层 数量不为零
        if len(stratumList) != 0:
            return smallModel(stratumList, dZone)
        else:
            return None











if __name__ == "__main__":
    def showLayerBoundary():
        lm = layerModelFromCIFLog("../otherThings/sPlot_演示12.scc")
        plt.style.use('_mpl-gallery')

        # make data
        x = lm.layerBoundaryLines[0].Y

        y = np.vstack(list(map(lambda lb: lb.Z, lm.layerBoundaryLines)))
        up = y.max()
        down = y.min()
        y[1:] = np.abs(y[:-1] - y[1:])

        # plot

        fig, ax = plt.subplots(figsize=(15, 10))
        fig.subplots_adjust(left=0.06, bottom=0.06)

        ax.stackplot(x, y,colors=['blue', 'orange',
                           'brown'], alpha=0.7)
        ax.invert_yaxis()
        ax.set(ylim=(up+10,down-10))
        plt.show()



    track = Trackofwell(r"C:\Users\41723\PycharmProjects\AutoMCNPInputGenerator\otherThings\HDRIFT.txt", r"C:\Users\41723\PycharmProjects\AutoMCNPInputGenerator\otherThings\VDRIFT.txt")
    lm = LayerModel("../otherThings/sPlot_演示12.scc", track)
    # for i in lm._CIFLogSCC.getroot():
    #     print(i)
    #     if i.tag == "LayerLines":
    #         for j in i:
    #             print("\t",j)
    #     if i.tag == "ControlLines":
    #         for j in i:
    #             print("\t",j)
    lm.t_getSmallModel(0)

    v = integrate.tplquad(volume_integral_function, -lm.radius, lm.radius, 0, 2 * np.pi, lm.radius, lm.radius)
    print(v)