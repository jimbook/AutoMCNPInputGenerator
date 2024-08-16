
from utility import geometry
from utility import coordinate

import numpy as np
from scipy import integrate
from tqdm import tqdm

from MCNPInput import MCNPAutoInput, GeometricModel, Material


from multiprocessing import Pool, cpu_count
# 建立一个边界面接口类
from abc import abstractmethod, ABCMeta

# 描述探测范围类
class DetectionZone(object):
    def __init__(self, relativeCoord: coordinate.CoordinateSystem, radius:float, innerDiameter:float):
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

    @property
    def all_volume(self) -> float:
        s = np.pi * (self.Radius * self.Radius - self.innerDiameter * self.innerDiameter)
        v = s * self.Radius * 2
        return v

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
        圆环边界描述的边界不应该作为判别地层界面的标准
        :param point:
        :return:
        '''
        # p = point.get_in_coord(self.zone.coord)
        # if p[1] < self.zone.Radius and p[1] > -self.zone.Radius:
        #     r = np.sqrt(p[0] * p[0] + p[2] * p[2])
        #     if r < self.zone.Radius and r > self.zone.innerDiameter:
        #         return True

        return True

    def _get_up_limit_line(self, y:float):
        return lambda theta: self.zone.Radius

    def _get_low_limit_line(self, y:float):
        return lambda theta: self.zone.innerDiameter

    def get_plane_plot_point(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return []

    def get_MCNP_Surface(self) -> list[tuple[GeometricModel.MCNP_UnionSurface | GeometricModel.MCNP_surface, int]]:
        columnSurface_out = GeometricModel.MCNP_surface('cy', [self.zone.Radius], note="detection boundary")
        columnSurface_inner = GeometricModel.MCNP_surface('cy', [self.zone.innerDiameter], note="borehole boundary")
        up_surface = GeometricModel.MCNP_PlaneSurface('py', [self.zone.Radius], note="Stratum detection upper interface")
        down_surface = GeometricModel.MCNP_PlaneSurface('py', [-self.zone.Radius], note="Stratum detection lower interface")
        return [(columnSurface_out, -1), (columnSurface_inner, 1), (up_surface, -1), (down_surface, 1)]

class OnePlaneBoundary(Boundary):
    def __init__(self, plane: geometry.Plane, detectorZone:DetectionZone = None):
        self.zone = detectorZone
        self.plane = plane.get_plane_in_coord(self.zone.coord)

    def is_in_it(self, point:coordinate.Point) -> bool:
        return self.plane.isUnderIt(point)

    def setZone(self,zone:DetectionZone):
        self.zone = zone

    def _get_up_limit_line(self, y:float):
        # 先判断在这个平面中，此边界为积分上限还是积分下限
        o = coordinate.Point(0, y, 0, coord=self.zone.coord)
        if self.is_in_it(o):  # 积分上限
            def line_function(theta: float):
                A, B, C, D = self.plane.analytic_equation_parameter
                # 如果平面垂直于y轴
                if B != 0 and A == 0 and C == 0:
                    # 对于上界情况，圆心在区域内，取全部面积，返回infinite，让上限为R
                    return np.Inf
                else:
                    w = A * np.cos(theta) + C * np.sin(theta)
                    # w == 0 说明直线过原点，这样就是说，当角度在一定范围内，rho为无穷大，其他时候为0
                    if np.isclose(w, 0):
                        # 计算一个在对应角度上的点
                        _p = np.array([np.cos(theta), y, np.sin(theta)])
                        _dotPN = np.dot(_p, self.plane.normal)
                        if _dotPN > 0:
                            return np.Inf
                        else:
                            return 0
                    else:
                        r = -(B * y - D) / w
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
                    # w == 0 说明直线过原点，这样就是说，当角度在一定范围内，rho为无穷大，其他时候为0
                    if np.isclose(w, 0):
                        # 计算一个在对应角度上的点
                        _p = np.array([np.cos(theta), y, np.sin(theta)])
                        _dotPN = np.dot(_p, self.plane.normal)
                        if _dotPN > 0:  # 在区域内
                            return 0
                        else:  # 在区域外
                            return np.Inf
                    else:
                        r = -(B * y - D) / w
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
        flag_x = np.dot(self.plane.normal, np.array([1, 0, 0]))
        flag_y = np.dot(self.plane.normal, np.array([0, 1, 0]))
        flag_z = np.dot(self.plane.normal, np.array([0, 0, 1]))
        if flag_x != 0:
            if flag_x > 0:
                return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), 1)]
            else:
                return [(GeometricModel.MCNP_PlaneSurface('p', param=(-self.plane).analytic_equation_parameter), -1)]
        elif flag_y != 0:
            if flag_y > 0:
                return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), 1)]
            else:
                return [(GeometricModel.MCNP_PlaneSurface('p', param=(-self.plane).analytic_equation_parameter), -1)]
        else:
            if flag_z > 0:
                return [(GeometricModel.MCNP_PlaneSurface('p', param=self.plane.analytic_equation_parameter), 1)]
            else:
                return [(GeometricModel.MCNP_PlaneSurface('p', param=(-self.plane).analytic_equation_parameter), -1)]

class UnionAndPlanesBoundary(Boundary):
    def __init__(self, boundary1:Boundary, boundary2:Boundary):
        self.b1 = boundary1
        self.b2 = boundary2

    def is_in_it(self, point:coordinate.Point) -> bool:
        return self.b1.is_in_it(point) and self.b2.is_in_it(point)

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

class stratum(object):
    '''
    地层描述类，能够自动计算自身体积
    注意，这里暂时不会检查坐标系是否统一
    '''
    def __init__(self, boundary: Boundary | None,
                 API:float, detectionZone: DetectionZone):
        '''

        :param topInterface: 地层上界面
        :param lowerInterface: 地层下界面
        :param API: 地层的API
        :param detectionZone: 探测区域
        '''
        # 将探测边界加入
        self.boundary:Boundary = boundary & AnnulusBoundary(detectionZone)
        self.api = API
        self.dZone = detectionZone
        # self._plot_model()

        self._volume = None
        self._volumeError = None

    @property
    def volume(self) -> float:
        if self._volume  is None:
            __tmp = self._calculate_volume()
            self._volume = __tmp[0]
            self._volumeError = __tmp[1]
        return self._volume

    # 计算体积
    def _calculate_volume_directly(self):

        # 进行三重积分了
        v = integrate.tplquad(volume_integral_function,  # 积分函数
                              -self.dZone.Radius, self.dZone.Radius,  # y轴的积分限
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
        v = integrate.quad(self._calculate_area, -self.dZone.Radius, self.dZone.Radius,  epsrel=5.0e-3)
        return v

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
        _y = np.linspace(-self.dZone.Radius, self.dZone.Radius, 100)
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
        _y = np.linspace(-self.dZone.Radius, self.dZone.Radius, 100)
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
        _y = np.linspace(-self.dZone.Radius, self.dZone.Radius, 1000)
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
        _y = np.linspace(-self.dZone.Radius, self.dZone.Radius, 1000)
        _theta = np.linspace(0, 2 * np.pi, 36 * 3)

        checkLimit = np.empty_like(_y)
        for i in tqdm(range(_y.shape[0]), postfix="check limit"):
            area_y: float = _y[i]
            p = coordinate.Point(0, area_y, 0, coord=self.dZone.coord)
            checkLimit[i] = 1 if self.boundary.is_in_it(p) else 0
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
            valmin=-self.dZone.Radius,
            valmax=self.dZone.Radius,
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
        h = np.linspace(-r, r, 160)
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

