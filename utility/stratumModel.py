import pandas as pd

import MCNPInput.SourceDefine
from utility import geometry
from utility import coordinate
from multiprocessing import cpu_count, Pool

import numpy as np
from scipy import integrate
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('_mpl-gallery')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


from MCNPInput import MCNPAutoInput, GeometricModel, Material, SourceDefine
# 建立一个边界面接口类
from abc import abstractmethod, ABCMeta

# 描述探测范围类
class DetectionZone(object):
    def __init__(self, relativeCoord: coordinate.CoordinateSystem, radius:float, innerDiameter:float,
                 topSurface:float = None, downSurface:float = None):
        '''
        描述探测范围的类，探测范围为相对坐标系下以原点为中心，以y轴为轴线的圆柱环区域
        :param relativeCoord: 探测区域的相对坐标系
        :param radius: 外径，单位cm
        :param innerDiameter:内径，单位cm
        '''
        self.coord = relativeCoord
        # 探测范围, cm
        self.Radius = radius # cm
        self.innerDiameter = innerDiameter
        self._allVolume = None
        # 额外参数
        if topSurface is None:
            self.topSurface = self.Radius
        else:
            self.topSurface = topSurface
        if downSurface is None:
            self.downSurface = -self.Radius
        else:
            self.downSurface = downSurface

    @property
    def all_volume(self) -> float:
        s = np.pi * (self.Radius * self.Radius - self.innerDiameter * self.innerDiameter)
        v = s * self.Radius * 2
        return v

    def get_SourceRange(self) -> dict[str:str|SourceDefine.MCNP_Distribution]:
        # 探测范围半径
        rad_si = SourceDefine.MCNP_SourceInformation()
        rad_sp = SourceDefine.MCNP_SourceProbabilty()
        rad_si.data.append(self.innerDiameter)
        rad_si.data.append(self.Radius)
        rad_sp.data.append(-21)
        rad_sp.data.append(1)
        rad = SourceDefine.MCNP_Distribution_discrete(rad_si, rad_sp)

        # 探测范围轴向
        ext_si = SourceDefine.MCNP_SourceInformation()
        ext_sp = SourceDefine.MCNP_SourceProbabilty()
        ext_si.data.append(self.downSurface)
        ext_si.data.append(self.topSurface)
        ext_sp.data.append(-21)
        ext_sp.data.append(0)
        ext = SourceDefine.MCNP_Distribution_discrete(ext_si, ext_sp)

        axs = '0 1 0'
        pos = '0 0 0'
        return {'rad':rad, "ext":ext, 'axs':axs, "pos":pos}

# 描述地层边界类
class Boundary(metaclass=ABCMeta):
    @abstractmethod
    def is_in_it(self, point:coordinate.Point) -> bool:
        pass

    @abstractmethod
    def _get_up_limit_line(self, y:float):
        pass

    @abstractmethod
    def _get_low_limit_line(self, y:float):
        pass

    def get_up_limit(self, y:float, theta:float) -> float:
        u = self._get_up_limit_line(y)(theta)
        return u

    def get_low_limit(self, y:float, theta:float) -> float:
        l = self._get_low_limit_line(y)(theta)
        u = self._get_up_limit_line(y)(theta)
        return min(l, u)

    @abstractmethod
    def get_plane_plot_point(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        pass

    @abstractmethod
    def get_MCNP_Surface(self) -> list[tuple[GeometricModel.MCNP_UnionSurface | GeometricModel.MCNP_surface, int]]:
        pass

    def __and__(self, other) -> 'Boundary':
        return UnionAndPlanesBoundary(self, other)

    def __or__(self, other) -> 'Boundary':
        return UnionOrPlanesBoundary(self, other)

class AnnulusBoundary(Boundary):
    def __init__(self, zone:DetectionZone):
        self.zone = zone

    def is_in_it(self, point:coordinate.Point) -> bool:
        '''
        圆环边界描述的边界,基于中心点判断上下边界不应该包含此
        :param point:
        :return:
        '''
        p = point.get_point_in_coord(self.zone.coord)
        if (p[1] < self.zone.topSurface) and (p[1] > self.zone.downSurface):
            r = np.sqrt(p[0] * p[0] + p[2] * p[2])
            if (r < self.zone.Radius) and (r > self.zone.innerDiameter):
                return True
        return False


    def _get_up_limit_line(self, y:float):
        return lambda theta: self.zone.Radius

    def _get_low_limit_line(self, y:float):
        return lambda theta: self.zone.innerDiameter

    def get_plane_plot_point(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return []

    def get_MCNP_Surface(self) -> list[tuple[GeometricModel.MCNP_UnionSurface | GeometricModel.MCNP_surface, int]]:
        columnSurface_out = GeometricModel.MCNP_surface('cy', [self.zone.Radius], note="detection boundary")
        columnSurface_inner = GeometricModel.MCNP_surface('cy', [self.zone.innerDiameter], note="borehole boundary")
        up_surface = GeometricModel.MCNP_PlaneSurface('py', [self.zone.topSurface], note="Stratum detection upper interface")
        down_surface = GeometricModel.MCNP_PlaneSurface('py', [self.zone.downSurface], note="Stratum detection lower interface")
        return [(columnSurface_out, -1), (columnSurface_inner, 1), (up_surface, -1), (down_surface, 1)]

class OnePlaneBoundary(Boundary):
    def __init__(self, plane: geometry.Plane, detectorZone:DetectionZone = None):
        self.zone = detectorZone
        if detectorZone is not None:
            self.plane = plane.get_plane_in_coord(self.zone.coord)
        else:
            self.plane = plane

    def is_in_it(self, point:coordinate.Point) -> bool:
        return self.plane.isUnderIt(point)

    def setZone(self,zone:DetectionZone):
        self.zone = zone

    def _get_up_limit_line(self, y:float):
        # 先判断在这个平面中，此边界为积分上限还是积分下限
        o = coordinate.Point(0, y, 0, coord=self.zone.coord)
        if self.is_in_it(o):  # 积分上限
            A, B, C, D = self.plane.analytic_equation_parameter
            def line_function(theta: float):
                # 如果平面垂直于y轴
                if B != 0 and A == 0 and C == 0:
                    # 对于上界情况，圆心在区域内，取全部面积，返回infinite，让上限为R
                    return np.Inf
                else:
                    # 当前y=y平面与其交线方程为：Ax+Cz+D_=0
                    # 转换为极坐标方程为 A * r * cos(theta) + C * r * sin(theta) + D_ = 0,
                    # 即：r * (A * cos(theta) + C * sin(theta)) + (B * y + D) = 0,
                    # r * w + (B * y + D) = 0
                    w = A * np.cos(theta) + C * np.sin(theta)
                    D_ = B * y + D
                    # w == 0, D_ == 0, 说明直线过原点，这样就是说，当角度在一定范围内，rho为无穷大，其他时候为0
                    if np.isclose(w, 0) and np.isclose(D_, 0):
                        # 计算一个在对应角度上的点
                        _p = np.array([np.cos(theta), y, np.sin(theta)])
                        _dotPN = np.dot(_p, self.plane.normal)
                        if _dotPN > 0:
                            return np.Inf
                        else:
                            return 0
                    elif np.isclose(w, 0):
                        return np.Inf
                    else:
                        r = -D_ / w
                        if r < 0:
                            return np.Inf
                        else:
                            return r
            return line_function
        else:  # 此边界为积分下限， 积分上限求最小，返回infinite来表示此边界无限制
            return lambda theta : np.Inf

    def _get_low_limit_line(self, y:float):
        # 先判断在这个平面中，此边界为积分上限还是积分下限
        o = coordinate.Point(0, y, 0, coord=self.zone.coord)
        if self.is_in_it(o): # 积分上限， 积分下限求最大， 返回0来表示此边界无限制
            return lambda x: 0
        else:
            def line_function(theta:float):
                A, B, C, D = self.plane.analytic_equation_parameter
                # 如果平面垂直于y轴
                if B != 0 and A == 0 and C == 0:
                    # 对于积分下限情况，圆心在区域外，返回无穷表示面积为0
                    return np.Inf
                else:
                    w = A * np.cos(theta) + C * np.sin(theta)
                    D_ = B * y + D
                    # w == 0 说明直线过原点，这样就是说，当角度在一定范围内，rho为无穷大，其他时候为0
                    if np.isclose(D_, 0) and np.isclose(w, 0):
                        # 计算一个在对应角度上的点
                        _p = np.array([np.cos(theta), y, np.sin(theta)])
                        _dotPN = np.dot(_p, self.plane.normal)
                        if _dotPN > 0:  # 在区域内
                            return 0
                        else:  # 在区域外
                            return np.Inf
                    elif np.isclose(w, 0):
                        return np.Inf
                    else:
                        r = -D_ / w
                        if r < 0:
                            return np.Inf
                        else:
                            return r
            return line_function

    def get_plane_plot_point(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        A, B, C, D = self.plane.analytic_equation_parameter
        xaxis = np.abs(np.dot(np.array([1, 0, 0]), self.plane.normal))
        yaxis = np.abs(np.dot(np.array([0, 1, 0]), self.plane.normal))
        zaxis = np.abs(np.dot(np.array([0, 0, 1]), self.plane.normal))
        max = np.max(np.array([xaxis, yaxis, zaxis]))
        r = self.zone.Radius
        if max == xaxis:
            _y = np.linspace(-r - 10, r + 10, 160)
            _z = np.linspace(-r - 10, r + 10, 160)
            y, z = np.meshgrid(_y, _z)
            x = (B * y + C * D) / -A
        elif max == zaxis:
            _x = np.linspace(-r - 10, r + 10, 160)
            _y = np.linspace(-r - 10, r + 10, 160)
            x, y = np.meshgrid(_x, _y)
            z = (A * x + B * y + D) / -C
        elif max == yaxis:
            _x = np.linspace(-r - 10, r + 10, 160)
            _z = np.linspace(-r - 10, r + 10, 160)
            x, z = np.meshgrid(_x, _z)
            y = (A * x + C * z + D) / -B
        return [(x, y ,z)]

    def get_MCNP_Surface(self) -> tuple[GeometricModel.MCNP_surface, int]:
        # flag_x = np.dot(self.plane.normal, self.plane.point0)
        # flag_y = np.dot(self.plane.normal, self.plane.point1)
        # flag_z = np.dot(self.plane.normal, self.plane.point2)

        # flag_vec = np.array([1,1,1])
        # flag = np.dot(flag_vec, self.plane.normal)
        # flag = int(flag / abs(flag))

        parameter = (*self.plane.analytic_equation_parameter[:3], -self.plane.analytic_equation_parameter[3])
        return [(GeometricModel.MCNP_PlaneSurface('p', param=parameter), 1)]
        #
        # if flag_x != 0:
        #     if flag_x > 0:
        #         return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), 1)]
        #     else:
        #         return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), -1)]
        # elif flag_y != 0:
        #     if flag_y > 0:
        #         return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), 1)]
        #     else:
        #         return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), -1)]
        # else:
        #     if flag_z > 0:
        #         return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), 1)]
        #     else:
        #         return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), -1)]

class UnionAndPlanesBoundary(Boundary):
    def __init__(self, boundary1:Boundary, boundary2:Boundary):
        self.b1 = boundary1
        self.b2 = boundary2

    def is_in_it(self, point:coordinate.Point) -> bool:
        _i1 = self.b1.is_in_it(point)
        _i2 = self.b2.is_in_it(point)
        return _i1 and _i2

    def _get_up_limit_line(self, y:float):
        line1 = self.b1._get_up_limit_line(y)
        line2 = self.b2._get_up_limit_line(y)
        def line_function(theta:float):
            a1 = line1(theta)
            a2 = line2(theta)
            a = min(a1, a2)
            return a
        return line_function

    def _get_low_limit_line(self, y:float):
        line1 = self.b1._get_low_limit_line(y)
        line2 = self.b2._get_low_limit_line(y)

        def line_function(theta: float):
            a1 = line1(theta)
            a2 = line2(theta)
            a = max(a1, a2)
            return a

        return line_function

    def get_plane_plot_point(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        l1 = self.b1.get_plane_plot_point()
        l2 = self.b2.get_plane_plot_point()
        l1.extend(l2)
        return l1

    def get_MCNP_Surface(self) -> list[tuple[GeometricModel.MCNP_UnionSurface | GeometricModel.MCNP_surface, int]]:
        s1 = self.b1.get_MCNP_Surface()
        s2 = self.b2.get_MCNP_Surface()
        s1.extend(s2)
        return s1

class UnionOrPlanesBoundary(Boundary):
    def __init__(self, boundary1:Boundary, boundary2:Boundary):
        self.b1 = boundary1
        self.b2 = boundary2

    def is_in_it(self, point:coordinate.Point) -> bool:
        return self.b1.is_in_it(point) or self.b2.is_in_it(point)

    def _get_up_limit_line(self, y:float):
        line1 = self.b1._get_up_limit_line(y)
        line2 = self.b2._get_up_limit_line(y)
        def line_function(theta:float):
            a1 = line1(theta)
            a2 = line2(theta)
            a = max(a1, a2)
            return a
        return line_function

    def _get_low_limit_line(self, y:float):
        line1 = self.b1._get_low_limit_line(y)
        line2 = self.b2._get_low_limit_line(y)

        def line_function(theta: float):
            a1 = line1(theta)
            a2 = line2(theta)
            a = min(a1, a2)
            return a

        return line_function

    def get_plane_plot_point(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        l1 = self.b1.get_plane_plot_point()
        l2 = self.b2.get_plane_plot_point()
        l1.extend(l2)
        return l1

    def get_MCNP_Surface(self) -> list[tuple[GeometricModel.MCNP_UnionSurface | GeometricModel.MCNP_surface, int]]:
        s1 = self.b1.get_MCNP_Surface()
        s2 = self.b2.get_MCNP_Surface()
        surfs = []
        for i in range(len(s1)):
            surf_i = s1[i]
            for j in range(len(s2)):
                surf_j = s2[j]
                surf_r = GeometricModel.MCNP_UnionSurface(surf_i, surf_j)
                surfs.append((surf_r, 1))
        return surfs

# 圆柱计算体积积分的方程
def volume_integral_function(rho, theta, y):
    # 由于scipy要求输入的函数依照func(z, y, x)，因此，对应改变参数排列
    return rho


def stratum_volume_multiprocessFunction(argList):
    stratum, downY, upY = argList
    v = integrate.quad(stratum._calculate_area, downY, upY, epsrel=5.0e-3)
    return v

# 描述一个地层
class Stratum(object):
    '''
    地层描述类，能够自动计算自身体积
    注意，这里暂时不会检查坐标系是否统一
    '''
    def __init__(self, boundary: Boundary | None,
                 API:float, detectionZone: DetectionZone, material:Material = None):
        '''

        :param topInterface: 地层上界面
        :param lowerInterface: 地层下界面
        :param API: 地层的API
        :param detectionZone: 探测区域
        :param material: 地层材料
        '''
        # 将探测边界加入
        self.boundary:Boundary = boundary & AnnulusBoundary(detectionZone)
        self._boundary = boundary
        self.api = API
        self.dZone = detectionZone
        # self._plot_model()
        self.material = material

        self._volume = None
        self._volumeError = None

    @property
    def volume(self) -> float:
        if self._volume  is None:
            __tmp = self._calculate_volume()
            self._volume = __tmp[0]
            self._volumeError = __tmp[1]
        return self._volume

    @property
    def doseStand(self) -> float:
        return self.api * self.volume

    def get_extRange(self, step:float = 1) -> tuple[float, float]:
        ext = np.arange(self.dZone.downSurface, self.dZone.topSurface, step)
        valid_idx = []
        for i in range(ext.shape[0]):
            extP = ext[i]
            area = self._calculate_area(extP)
            if not np.allclose(area, 0.):
                valid_idx.append(extP)
        valid_idx = np.array(valid_idx)
        valid_max = np.max(valid_idx)
        valid_min = np.min(valid_idx)

        return min(valid_max, self.dZone.topSurface), max(valid_min, self.dZone.downSurface)

    # 计算体积
    def _calculate_volume_directly(self):
        # 进行三重积分了
        v = integrate.tplquad(volume_integral_function,  # 积分函数
                              self.dZone.downSurface, self.dZone.topSurface,  # y轴的积分限
                              0., 2 * np.pi,  # theta的积分限
                              self.boundary.get_low_limit, self.boundary.get_up_limit,  # rho的积分限
                              epsrel=5.0e-3)
        return v

    # 计算对应截面下的面积
    def _calculate_area(self, y:float) -> float:
        line_up = self.boundary._get_up_limit_line(y)
        line_low = self.boundary._get_low_limit_line(y)
        def get_rho(theta:float):
            up_limit = line_up(theta)
            low_limit = line_low(theta)
            if up_limit <= low_limit:
                return 0
            else:
                rho = 0.5 * (up_limit * up_limit - low_limit * low_limit)
                return rho
        s = integrate.quad(get_rho, 0, 2*np.pi, epsrel=5.0e-3)
        return s[0]

    def _calculate_volume(self):
        v = integrate.quad(self._calculate_area, self.dZone.downSurface, self.dZone.topSurface,  epsrel=5.0e-3)
        return v

    def _calculate_volume_multiprocess(self):
        split_y = np.linspace(self.dZone.downSurface, self.dZone.topSurface, num=cpu_count() + 1)
        argList = []
        for i in range(cpu_count()):
            argList.append((self, split_y[i], split_y[i+1]))
        with Pool(cpu_count()) as p:
            result = p.map(stratum_volume_multiprocessFunction, argList)
        result_np = np.array(result)
        v = np.sum(result_np[:, 0])
        e = np.sum(result_np[:, 1])
        return v,e


    def _plot_limit(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=plt.figaspect(0.5))



        # ax = fig.add_subplot(2, 3, 1)
        # self._addPlot_areaOfSplit(fig,ax)
        # ax = fig.add_subplot(2, 3, 2)
        # self._addPlot_checkLimit(fig, ax)
        #
        # ax = fig.add_subplot(2, 3, 3, projection='polar')
        # self._addPlot_eachSplit(fig,ax)

        # ax = fig.add_subplot(2, 3, 4, projection='3d')
        # self._addPlot_3DModel(fig, ax)
        #
        ax = fig.add_subplot(1, 2, 1)
        self._addPlot_uplimit(fig,ax)

        ax = fig.add_subplot(1, 2, 2)
        self._addPlot_lowlimit(fig, ax)

        # fig.tight_layout()

        plt.show()

    def _plot_model(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        self._addPlot_3DModel(fig, ax)
        plt.show()

    def _plot_area(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 3, 1)
        self._addPlot_areaOfSplit(fig,ax)
        ax = fig.add_subplot(1, 3, 2)
        self._addPlot_checkLimit(fig, ax)
        ax = fig.add_subplot(1, 3, 3, projection='polar')
        self._addPlot_eachSplit(fig,ax)

        plt.show()

    def _addPlot_uplimit(self,fig, ax):
        _y = np.linspace(self.dZone.downSurface, self.dZone.topSurface, 100)
        _theta = np.linspace(0, 2 * np.pi, 36 * 3)

        y, theta = np.meshgrid(_y, _theta)
        theta_angle = theta / np.pi * 180

        upL = np.empty_like(y)
        pbar = tqdm(total=y.shape[0] * y.shape[1], postfix="up limit")
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                upL[i, j] = self.boundary.get_up_limit(y[i, j], theta[i, j])
                pbar.update(1)

        im = ax.pcolormesh(y, theta_angle, upL, shading='gouraud')
        ax.set_title("up limit")
        ax.grid(True)
        fig.colorbar(im, ax=ax)

    def _addPlot_lowlimit(self, fig, ax):
        _y = np.linspace(self.dZone.downSurface, self.dZone.topSurface, 100)
        _theta = np.linspace(0, 2 * np.pi, 36 * 3)

        y, theta = np.meshgrid(_y, _theta)
        theta_angle = theta / np.pi * 180

        downL = np.zeros_like(y)
        pbar = tqdm(total=y.shape[0] * y.shape[1], postfix="low limit")
        for j in range(y.shape[1]):
            for i in range(y.shape[0]):
                downL[i, j] = self.boundary.get_low_limit(y[i, j], theta[i, j])
                pbar.update(1)

        im = ax.pcolormesh(y, theta_angle, downL, shading='gouraud')
        ax.set_title("low limit")
        ax.grid(True)
        fig.colorbar(im, ax=ax)

    def _addPlot_areaOfSplit(self, fig, ax):
        _y = np.linspace(self.dZone.downSurface, self.dZone.topSurface, 1000)
        _theta = np.linspace(0, 2 * np.pi, 36 * 3)
        area = np.empty_like(_y)
        for i in tqdm(range(_y.shape[0]), postfix="area"):
            area_y:float = _y[i]
            s = self._calculate_area(area_y)
            area[i] = s
        im = ax.plot(_y, area)
        ax.set_title("area")
        ax.grid(True)

    def _addPlot_checkLimit(self,fig, ax):
        _y = np.linspace(self.dZone.downSurface, self.dZone.topSurface, 1000)
        _theta = np.linspace(0, 2 * np.pi, 36 * 3)

        checkLimit = np.empty_like(_y)
        for i in tqdm(range(_y.shape[0]), postfix="check limit"):
            area_y: float = _y[i]
            p = coordinate.Point(0, area_y, 0, coord=self.dZone.coord)
            checkLimit[i] = 1 if self._boundary.is_in_it(p) else 0
        im = ax.plot(_y, checkLimit)
        ax.set_title("position of limit")
        ax.grid(True)

    def _addPlot_eachSplit(self,fig, ax):
        from matplotlib.widgets import Slider
        fig.subplots_adjust(right=0.85, bottom=0.25)
        y = 10
        theta = np.linspace(0, 2 * np.pi, 360)
        rho_up_f = self.boundary._get_up_limit_line(y)
        rho_low_f = self.boundary._get_low_limit_line(y)
        rho_up = np.empty_like(theta)
        rho_low = np.empty_like(theta)
        for i in range(theta.shape[0]):
            rho_up[i] = rho_up_f(theta[i])
            rho_low[i] = rho_low_f(theta[i])
        self._line_up, = ax.plot(theta, rho_up, 'y')
        self._line_low, = ax.plot(theta, rho_low, 'b')
        ax.set_ylim(0, self.dZone.Radius * 1.1)
        ax.grid(True)
        ax.set_title("split")

        axfreq = fig.add_axes([0.95, 0.25, 0.0225, 0.63])
        self._freq_slider = Slider(
            ax=axfreq,
            label='y',
            valmin=self.dZone.downSurface,
            valmax=self.dZone.topSurface,
            valinit=y,
            orientation="vertical"
        )
        def update(val):
            rho_up_f = self.boundary._get_up_limit_line(val)
            rho_low_f = self.boundary._get_low_limit_line(val)
            rho_up = np.empty_like(theta)
            rho_low = np.empty_like(theta)
            for i in range(theta.shape[0]):
                rho_up[i] = rho_up_f(theta[i])
                rho_low[i] = rho_low_f(theta[i])
                if rho_low[i] >= rho_up[i]:
                    rho_low[i] = rho_up[i]
            self._line_up.set_ydata(rho_up)
            self._line_low.set_ydata(rho_low)
            fig.canvas.draw_idle()

        self._freq_slider.on_changed(update)
        self._freq_slider.reset()

    def _addPlot_3DModel(self,fig, ax):
        import matplotlib as mpl
        r = self.dZone.Radius
        u = np.linspace(0, 2 * np.pi, 360)
        h = np.linspace(self.dZone.downSurface, self.dZone.topSurface, 160)
        x = np.outer(np.cos(u) * r, np.ones(len(h)))
        y = np.outer(np.ones(len(u)), h)
        z = np.outer(np.sin(u) * r, np.ones(len(h)))
        theta = np.outer(u, np.ones(len(h)))

        # viridis = mpl.colormaps['viridis'].resampled(8)
        # _chT = np.linspace(-2, 2, 41)
        # _chT_color = viridis(_chT)

        downL = np.zeros_like(y)
        pbar = tqdm(total=y.shape[1] * y.shape[0], postfix="3D down")
        for j in range(y.shape[1]):
            for i in range(y.shape[0]):
                downL[i, j] = self.boundary.get_low_limit(y[i, j], theta[i, j])
                pbar.update(1)
        pbar.close()

        upL = np.empty_like(y)
        pbar = tqdm(total=y.shape[0] * y.shape[1], postfix="3D up")
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                upL[i, j] = self.boundary.get_up_limit(y[i, j], theta[i, j])
                pbar.update(1)
        pbar.close()

        rho = upL - downL
        # Plot the surface
        viridis = mpl.colormaps['viridis'].resampled(8)
        rho_toOne = (rho-np.min(rho))/(np.max(rho)-np.min(rho))
        color_rho = viridis(rho_toOne)
        color_rho[:, :, 3] = 0.7
        ax.plot_surface(x, y, z, facecolors=color_rho)
        co_list = self.boundary.get_plane_plot_point()
        for x, y, z in co_list:
            ax.plot_surface(x, y, z)
        ax.axis('equal')



# 井眼轨迹类
class WellTrack(object):
    def __init__(self):
        self.trackPD = pd.DataFrame(columns=["Depth", "X", "Y", "Z", "Inc", "Azi"]).set_index("Depth")
        self.startDepth = 0

    @property
    def track(self) -> pd.DataFrame:
        return self.trackPD.iloc[self.startDepth:]

    def __getitem__(self, item):
        return self.trackPD.iloc[self.startDepth + item]

    def __len__(self):
        return self.trackPD.shape[0] - self.startDepth

# 描述一次探测的地层情况
class SingleStratumModel(object):
    def __init__(self, stratum_border: list[Stratum], zone: DetectionZone, **kwargs):
        self.stratumBorders = stratum_border
        self.zone = zone
        self.kwargs = kwargs

    def get_total_doseStand(self):
        _d = 0.
        for s in self.stratumBorders:
            _d += s.doseStand
        return _d

    def get_valid_volume(self):
        _v = 0.
        for s in self.stratumBorders:
            _v += s.volume
        return _v

    def __getitem__(self, item):
        return self.stratumBorders[item]

    def __len__(self):
        return len(self.stratumBorders)

    def __getattr__(self, item):
        return self.kwargs.get(item, None)


# 描述整个地层模型
class StratumModel(metaclass=ABCMeta):
    def __init__(self, trajectory: WellTrack, radius_of_investigation:float, well_diameter: float):
        self.trajectory = trajectory
        self.radius = radius_of_investigation
        self.diameter = well_diameter

    @abstractmethod
    def __getitem__(self, item) -> SingleStratumModel:
        pass

    @abstractmethod
    def __len__(self):
        pass

# 用pd.DataFrame来存储地层模型描述
class StratumModel_DataFrame(StratumModel):
    def __init__(self, stratumData:pd.DataFrame, api:np.array, trajectory: WellTrack,
                 radius_of_investigation:float, well_diameter: float,
                 stratumAzi:np.array = None):
        super().__init__(trajectory, radius_of_investigation, well_diameter)
        self.api = api
        self.data = stratumData # watch T.R.
        self.Azi = stratumAzi # 方位角
        # 检查数据匹配
        if self.api.shape[0] != self.data.shape[1]:
            raise Exception("self.api.shape[0] != self.data.shape[1]")

        if self.Azi is not None:
            if self.Azi.shape[0] != self.data.shape[0] - 1 or self.Azi.shape[1] != self.data.shape[1]:
                raise Exception("Azi doesn't match to stratum data")

        # 根据井眼轨迹充填地层模型左右边界
        # -获取轨迹以及地层左右最值
        track_x = self.trajectory.track["X"].values
        track_x_min = np.min(track_x) - self.radius * 2
        track_x_max = np.max(track_x) + self.radius * 2
        layer_x = self.data.index.values
        layer_x_min = np.min(layer_x)
        layer_x_max = np.max(layer_x)
        layer_x_list = layer_x.tolist()
        layer_data = self.data.values
        change_flag = False

        #
        if track_x_min < layer_x_min:
            tmp_y = layer_data[0]
            layer_data = np.vstack([tmp_y, *layer_data])
            layer_x_list.insert(0, track_x_min)
            # 如果有角度数据
            if self.Azi is not None:
                self.Azi = np.insert(self.Azi, 0, self.Azi[0], axis=0)
            change_flag = True
        if track_x_max > layer_x_max:
            tmp_y = layer_data[-1]
            layer_data = np.vstack([*layer_data, tmp_y])
            layer_x_list.append(track_x_max)
            # 如果有角度数据
            if self.Azi is not None:
                self.Azi = np.insert(self.Azi, -1, self.Azi[1], axis=0)
            change_flag = True
        if change_flag:
            self.data = pd.DataFrame(layer_data, index=layer_x_list)
        # 根据井眼轨迹筛去离地层垂直距离过远的测量点
        y_min = np.min(layer_data) - self.radius * 2
        for i in range(len(self.trajectory)):
            track_y = self.trajectory.track["Y"].values[i]
            if y_min < track_y:
                self.trajectory.startDepth += max(i - 1, 0)
                break

    def getSmallModel(self, index:int) -> SingleStratumModel:
        logger.info("get small model {:d}".format(index))
        if index < 0 or index >= self.trajectory.track.shape[0]:
            return None
        if index >= 28:
            _tmpFlag = " "
        # 探测点坐标
        p = self.trajectory.track.iloc[index]
        d_x = p["X"]
        d_y = p["Y"]
        d_z = p["Z"]
        d_inc = p["Inc"]
        # 建立一个探测器相对坐标系
        coord = (coordinate.CoordinateSystem().move_to(coordinate.ThreeDVector(d_x, d_y, d_z)).
                 rotate(coordinate.ThreeDVector(0,0,1),
                        coordinate.radian_to_angle(-d_inc)))
        dZone = DetectionZone(coord, self.radius, self.diameter)
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
        _x_array = self.data.index.values
        index_x_min = _x_array.searchsorted(x_min) - 1
        index_x_max = _x_array.searchsorted(x_max)
        logger.info("\t x range: {:d}({:f}) ~ {:d}({:f})".format(index_x_min, _x_array[index_x_min], index_x_max, _x_array[index_x_max]))
        # 如果探测范围中穿过了一条以上的控制线，则抛出地层界面变化过快的异常
        if index_x_max - index_x_min > 2:
            raise Exception("The stratum interface changes rapidly.")


        # 一个地层的边界线是上界线， api是给的对应边界下面的一个地层的值， 所以只有下边界无穷的地层
        # 第一个地层上边界
        upBoundary = []
        for i in range(index_x_min, index_x_max):
            controlPoint_0 = coordinate.Point(self.data.index.values[i], self.data.iloc[i, 0], 0)
            controlPoint_1 = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, 0], 0)
            if self.Azi is None:
                # 使用右手系，上界面方向向量应该y轴正方向
                dipDirectionPoint = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, 0], 1)
            else:
                # 如果有倾向数据，则根据倾向来确定点，注意上界平面会被反向
                azi = self.Azi[i, 0]
                _centrePoint = coordinate.ThreeDVector(self.data.index.values[i + 1], self.data.iloc[i + 1, 0], 0)
                _tmpCoord = coordinate.CoordinateSystem().move_to(_centrePoint).rotate(coordinate.ThreeDVector(0,1,0), azi)
                dipDirectionPoint = coordinate.Point(0, 0, 1, _tmpCoord).get_point_in_origin()

            plane = -geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint, coord)
            upBoundary.append(plane)

        stratumList = []
        for j in range(1, self.data.shape[1] + 1):
            # 这个循环内是计算一个地层的，循环每一个地层
            # 下界面
            lowerBoundary = []
            if j == self.data.shape[1]:
                lowerBoundary = None
            else:
                #建立上下界面
                for i in range(index_x_min, index_x_max):
                    controlPoint_0 = coordinate.Point(self.data.index.values[i], self.data.iloc[i, j], 0)
                    controlPoint_1 = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, j], 0)
                    if self.Azi is None:
                        # 使用右手系，下界面方向向量应该y轴负方向，z应为正
                        dipDirectionPoint = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, j], 1)
                    else:
                        # 如果有倾向数据，则根据倾向来确定点
                        azi = self.Azi[i, j]
                        _centrePoint = coordinate.ThreeDVector(self.data.index.values[i + 1], self.data.iloc[i + 1, j], 0)
                        _tmpCoord = coordinate.CoordinateSystem().move_to(_centrePoint).rotate(
                            coordinate.ThreeDVector(0, 1, 0), azi)
                        dipDirectionPoint = coordinate.Point(0, 0, 1, _tmpCoord).get_point_in_origin()

                    # 注意转换坐标系
                    plane = geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint,coord)
                    lowerBoundary.append(plane)


            if index >= 28:
                qrien = "ds"
            # 先使用非体积方法排除一些地层
            # 上下界面限制, 这里在面对具有倾向的地层界面可能会导致被错误排除，这里在存在倾向时禁止这个判断
            if self.Azi is None:
                upData = self.data.iloc[index_x_min:index_x_max + 1, j - 1]
                layer_min = np.min(upData)
                if y_max < layer_min:
                    continue
                if lowerBoundary is not None:
                    downData = self.data.iloc[index_x_min:index_x_max + 1, j]
                    layer_max = np.max(downData)
                    if y_min > layer_max:
                        continue
            # TEST://
            #建立地层对象
            if lowerBoundary is None:
                pass
            elif len(lowerBoundary) == 1:
                lowB = OnePlaneBoundary(lowerBoundary[0], dZone)
            elif len(lowerBoundary) == 2:
                lowB_1 = OnePlaneBoundary(lowerBoundary[0], dZone)
                lowB_2 = OnePlaneBoundary(lowerBoundary[1], dZone)
                low_crossVec = np.cross(lowerBoundary[0].normal, lowerBoundary[1].normal)
                if np.dot(low_crossVec, np.array([0, 0, 1])) > 0:
                    lowB = UnionOrPlanesBoundary(lowB_1, lowB_2)
                else:
                    lowB = UnionAndPlanesBoundary(lowB_1, lowB_2)
            if len(upBoundary) == 1:
                upB = OnePlaneBoundary(upBoundary[0], dZone)
            elif len(upBoundary) == 2:
                upB_1 = OnePlaneBoundary(upBoundary[0], dZone)
                upB_2 = OnePlaneBoundary(upBoundary[1], dZone)
                up_crossVec = np.cross(upBoundary[0].normal, upBoundary[1].normal)
                if np.dot(up_crossVec, np.array([0, 0, 1])) > 0:
                    upB = UnionAndPlanesBoundary(upB_1, upB_2)
                else:
                    upB = UnionOrPlanesBoundary(upB_1, upB_2)

            if lowerBoundary is not None:
                B = UnionAndPlanesBoundary(upB, lowB)
            else:
                B = upB
            stratum = Stratum(B, self.api[j-1], dZone)
            v = stratum.volume
            if not np.isclose(v, 0):
                stratumList.append(stratum)
                # stratum._plot_area()
            # 下界面将会变成下一个地层的上界面，需要将界面的法向量反转
            # 如果没有指定下界面，那么应为最后一个地层，不在需要指定下一地层的上界面
            if lowerBoundary is not None:
                upBoundary = list(map(lambda x: -x, lowerBoundary))

        #如果 体积不等于零的地层 数量不为零
        if len(stratumList) != 0:
            return SingleStratumModel(stratumList, dZone)
        else:
            return None

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, item) -> SingleStratumModel:
        return self.getSmallModel(item)

    def plot(self):
        _data = self.data.values.T
        _data_stack = _data[1:] - _data[:-1]
        bottom = np.min(_data[0])
        y = np.vstack([_data[0], *_data_stack])
        x = self.data.index.values.astype(float)
        track_x = self.trajectory.track["X"].values
        track_y = self.trajectory.track["Y"].values

        _cmap = mpl.colormaps['viridis']
        _api_min = np.nanmin(self.api,axis=0)
        _api_max = np.nanmax(self.api,axis=0)
        _api_normal = mpl.colors.Normalize(vmin=_api_min, vmax=_api_max)
        _colorMap = mpl.cm.ScalarMappable(_api_normal, _cmap)
        _stratumColor = _colorMap.to_rgba(self.api)

        # plot
        fig, _ax = plt.subplots(2, 1, height_ratios=[100, 1])
        ax = _ax[0]
        ax.stackplot(x, y, colors=_stratumColor)
        fig.colorbar(_colorMap, cax = _ax[1],orientation='horizontal')

        ax.plot(track_x, track_y,'r')

        ax.invert_yaxis()
        ax.set(xlim=(min(np.min(x), np.min(track_x)), max(np.max(x), np.max(track_x))),
               ylim=(np.max(_data), np.min(_data)))
        plt.show()

class StratumModel_DataFrame_complete(StratumModel):
    def __init__(self, stratumData:pd.DataFrame, api:np.array, trajectory: WellTrack,
                 radius_of_investigation:float, well_diameter: float,
                 topSurface:float, downSurface:float,
                 stratumAzi:np.array = None):
        super().__init__(trajectory, radius_of_investigation, well_diameter)
        self.api = api
        self.data = stratumData # watch T.R.
        self.Azi = stratumAzi # 方位角
        self.topSurface = topSurface
        self.downSurface = downSurface
        # 检查数据匹配
        if self.api.shape[0] != self.data.shape[1]:
            raise Exception("self.api.shape[0] != self.data.shape[1]")

        if self.Azi is not None:
            if self.Azi.shape[0] != self.data.shape[0] - 1 or self.Azi.shape[1] != self.data.shape[1]:
                raise Exception("Azi doesn't match to stratum data")

        # 根据井眼轨迹充填地层模型左右边界
        # -获取轨迹以及地层左右最值
        track_x = self.trajectory.track["X"].values
        track_x_min = np.min(track_x) - self.radius * 2
        track_x_max = np.max(track_x) + self.radius * 2
        layer_x = self.data.index.values
        layer_x_min = np.min(layer_x)
        layer_x_max = np.max(layer_x)
        layer_x_list = layer_x.tolist()
        layer_data = self.data.values
        change_flag = False

        #
        if track_x_min < layer_x_min:
            tmp_y = layer_data[0]
            layer_data = np.vstack([tmp_y, *layer_data])
            layer_x_list.insert(0, track_x_min)
            # 如果有角度数据
            if self.Azi is not None:
                self.Azi = np.insert(self.Azi, 0, self.Azi[0], axis=0)
            change_flag = True
        if track_x_max > layer_x_max:
            tmp_y = layer_data[-1]
            layer_data = np.vstack([*layer_data, tmp_y])
            layer_x_list.append(track_x_max)
            # 如果有角度数据
            if self.Azi is not None:
                self.Azi = np.insert(self.Azi, -1, self.Azi[1], axis=0)
            change_flag = True
        if change_flag:
            self.data = pd.DataFrame(layer_data, index=layer_x_list)
        # 根据井眼轨迹筛去离地层垂直距离过远的测量点
        y_min = np.min(layer_data) - self.radius * 2
        for i in range(len(self.trajectory)):
            track_y = self.trajectory.track["Y"].values[i]
            if y_min < track_y:
                self.trajectory.startDepth += max(i - 1, 0)
                break

    def getSmallModel(self, index:int) -> SingleStratumModel:
        logger.info("get small model {:d}".format(index))
        if index < 0 or index >= self.trajectory.track.shape[0]:
            return None
        if index >= 28:
            _tmpFlag = " "
        # 探测点坐标
        p = self.trajectory.track.iloc[index]
        d_x = p["X"]
        d_y = p["Y"]
        d_z = p["Z"]
        d_inc = p["Inc"]
        # 建立一个探测器相对坐标系
        coord = (coordinate.CoordinateSystem().move_to(coordinate.ThreeDVector(d_x, d_y, d_z)).
                 rotate(coordinate.ThreeDVector(0,0,1),
                        coordinate.radian_to_angle(-d_inc)))
        dZone = DetectionZone(coord, self.radius, self.diameter, topSurface=self.topSurface, downSurface=self.downSurface)
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
        _x_array = self.data.index.values
        index_x_min = _x_array.searchsorted(x_min) - 1
        index_x_max = _x_array.searchsorted(x_max)
        logger.info("\t x range: {:d}({:f}) ~ {:d}({:f})".format(index_x_min, _x_array[index_x_min], index_x_max, _x_array[index_x_max]))
        # 如果探测范围中穿过了一条以上的控制线，则抛出地层界面变化过快的异常
        if index_x_max - index_x_min > 2:
            raise Exception("The stratum interface changes rapidly.")


        # 一个地层的边界线是上界线， api是给的对应边界下面的一个地层的值， 所以只有下边界无穷的地层
        # 第一个地层上边界
        upBoundary = []
        for i in range(index_x_min, index_x_max):
            controlPoint_0 = coordinate.Point(self.data.index.values[i], self.data.iloc[i, 0], 0)
            controlPoint_1 = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, 0], 0)
            if self.Azi is None:
                # 使用右手系，上界面方向向量应该y轴正方向
                dipDirectionPoint = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, 0], 1)
            else:
                # 如果有倾向数据，则根据倾向来确定点，注意上界平面会被反向
                azi = self.Azi[i, 0]
                _centrePoint = coordinate.ThreeDVector(self.data.index.values[i + 1], self.data.iloc[i + 1, 0], 0)
                _tmpCoord = coordinate.CoordinateSystem().move_to(_centrePoint).rotate(coordinate.ThreeDVector(0,1,0), azi)
                dipDirectionPoint = coordinate.Point(0, 0, 1, _tmpCoord).get_point_in_origin()

            plane = -geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint, coord)
            upBoundary.append(plane)

        stratumList = []
        for j in range(1, self.data.shape[1] + 1):
            # 这个循环内是计算一个地层的，循环每一个地层
            # 下界面
            lowerBoundary = []
            if j == self.data.shape[1]:
                lowerBoundary = None
            else:
                #建立上下界面
                for i in range(index_x_min, index_x_max):
                    controlPoint_0 = coordinate.Point(self.data.index.values[i], self.data.iloc[i, j], 0)
                    controlPoint_1 = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, j], 0)
                    if self.Azi is None:
                        # 使用右手系，下界面方向向量应该y轴负方向，z应为正
                        dipDirectionPoint = coordinate.Point(self.data.index.values[i + 1], self.data.iloc[i + 1, j], 1)
                    else:
                        # 如果有倾向数据，则根据倾向来确定点
                        azi = self.Azi[i, j]
                        _centrePoint = coordinate.ThreeDVector(self.data.index.values[i + 1], self.data.iloc[i + 1, j], 0)
                        _tmpCoord = coordinate.CoordinateSystem().move_to(_centrePoint).rotate(
                            coordinate.ThreeDVector(0, 1, 0), azi)
                        dipDirectionPoint = coordinate.Point(0, 0, 1, _tmpCoord).get_point_in_origin()

                    # 注意转换坐标系
                    plane = geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint,coord)
                    lowerBoundary.append(plane)

            # TEST://
            #建立地层对象
            if lowerBoundary is None:
                pass
            elif len(lowerBoundary) == 1:
                lowB = OnePlaneBoundary(lowerBoundary[0], dZone)
            elif len(lowerBoundary) == 2:
                lowB_1 = OnePlaneBoundary(lowerBoundary[0], dZone)
                lowB_2 = OnePlaneBoundary(lowerBoundary[1], dZone)
                low_crossVec = np.cross(lowerBoundary[0].normal, lowerBoundary[1].normal)
                if np.dot(low_crossVec, np.array([0, 0, 1])) > 0:
                    lowB = UnionOrPlanesBoundary(lowB_1, lowB_2)
                else:
                    lowB = UnionAndPlanesBoundary(lowB_1, lowB_2)
            if len(upBoundary) == 1:
                upB = OnePlaneBoundary(upBoundary[0], dZone)
            elif len(upBoundary) == 2:
                upB_1 = OnePlaneBoundary(upBoundary[0], dZone)
                upB_2 = OnePlaneBoundary(upBoundary[1], dZone)
                up_crossVec = np.cross(upBoundary[0].normal, upBoundary[1].normal)
                if np.dot(up_crossVec, np.array([0, 0, 1])) > 0:
                    upB = UnionAndPlanesBoundary(upB_1, upB_2)
                else:
                    upB = UnionOrPlanesBoundary(upB_1, upB_2)

            if lowerBoundary is not None:
                B = UnionAndPlanesBoundary(upB, lowB)
            else:
                B = upB
            stratum = Stratum(B, self.api[j-1], dZone)
            v = stratum.volume
            if not np.isclose(v, 0):
                stratumList.append(stratum)
                # stratum._plot_area()
            # 下界面将会变成下一个地层的上界面，需要将界面的法向量反转
            # 如果没有指定下界面，那么应为最后一个地层，不在需要指定下一地层的上界面
            if lowerBoundary is not None:
                upBoundary = list(map(lambda x: -x, lowerBoundary))

        #如果 体积不等于零的地层 数量不为零
        if len(stratumList) != 0:
            return SingleStratumModel(stratumList, dZone)
        else:
            return None

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, item) -> SingleStratumModel:
        return self.getSmallModel(item)

    def plot(self):
        _data = self.data.values.T
        _data_stack = _data[1:] - _data[:-1]
        bottom = np.min(_data[0])
        y = np.vstack([_data[0], *_data_stack])
        x = self.data.index.values.astype(float)
        track_x = self.trajectory.track["X"].values
        track_y = self.trajectory.track["Y"].values

        _cmap = mpl.colormaps['viridis']
        _api_min = np.nanmin(self.api,axis=0)
        _api_max = np.nanmax(self.api,axis=0)
        _api_normal = mpl.colors.Normalize(vmin=_api_min, vmax=_api_max)
        _colorMap = mpl.cm.ScalarMappable(_api_normal, _cmap)
        _stratumColor = _colorMap.to_rgba(self.api)

        # plot
        fig, _ax = plt.subplots(2, 1, height_ratios=[100, 1])
        ax = _ax[0]
        ax.stackplot(x, y, colors=_stratumColor)
        fig.colorbar(_colorMap, cax = _ax[1],orientation='horizontal')

        ax.plot(track_x, track_y,'r')

        ax.invert_yaxis()
        ax.set(xlim=(min(np.min(x), np.min(track_x)), max(np.max(x), np.max(track_x))),
               ylim=(np.max(_data), np.min(_data)))
        plt.show()
