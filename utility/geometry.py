import copy

import numpy as np

from .coordinate import *

class Plane(object):
    def __init__(self, point0: Point, point1: Point, point2: Point):
        self.coord = point0.coord
        self.point0 = point0
        self.point1 = point1.get_in_coord(coord=self.coord)
        self.point2 = point2.get_in_coord(coord=self.coord)
        self.normal = self._normal()
        self.analytic_equation_parameter = self._analytic_equation_parameter()

    def _normal(self):
        v1 = self.point1 - self.point0
        v2 = self.point2 - self.point1
        normal = np.cross(v1, v2)
        n = normal / np.linalg.norm(normal)
        return n # 法向量

    def isUnderIt(self, point:Point) -> bool:
        '''
        检查点是否在平面划分的区域内
        :param point:
        :return:
        '''
        if point.coord == self.coord:
            v = point - self.point0
            if np.isclose(np.linalg.norm(v), 0):
                v = point - self.point1
            a = np.dot(v, self.normal)
            if np.isclose(a, 0):
                return True
            elif a > 0:
                return True
            else:
                return False
        else:
            return self.get_plane_in_coord(point.coord).isUnderIt(point)

    # def isInIt(self, point:Point) -> bool:
    #     '''
    #     检查点是否在平面上
    #     :param point:
    #     :return:
    #     '''

    def _analytic_equation_parameter(self) -> tuple[float]:
        '''
        给出平面解析方程的四个参数，方程形式为Ax+By+Cz+D = 0
        :return: A, B, C, D
        '''
        A = self.normal[0]
        B = self.normal[1]
        C = self.normal[2]
        D = -np.dot(self.point0, self.normal)
        return A, B, C, D

    def get_plane_in_coord(self,coord: CoordinateSystem | None):
        if coord is None:
            coord = CoordinateSystem()
        p0 = self.point0.get_in_coord(coord)
        p1 = self.point1.get_in_coord(coord)
        p2 = self.point2.get_in_coord(coord)
        return Plane(p0, p1, p2)

    def integral_boundary(self,y: float, theta: float, coord:CoordinateSystem, upper:bool):
        A, B, C, D = self.analytic_equation_parameter
        # 如果平面垂直于y轴
        if B != 0 and A == 0 and C == 0:
            n = self.normal
            p = self.point0 - Point(0, y, 0, coord)
            d = np.dot(n, p)
            # 如果圆心在区域内，此界限为积分上界；如果圆心在区域外，此界限为下界
            # 对于上界情况，返回None，让上限为R
            # 对于下界情况，返回无穷大，这样积分下界超过上界时，上层函数会将面积置零
            if d > 0:
                return np.Inf
            else:
                return None
        else:
            w = A * np.cos(theta) + C * np.sin(theta)
            # w == 0 说明直线过原点，这样就是说，当角度在一定范围内，rho为无穷大，其他时候为0
            if np.isclose(w, 0):
                # 计算一个在对应角度上的点
                _p = Point(np.cos(theta), y, np.sin(theta))
                _dotPN = np.dot(_p, self.normal)
                if _dotPN > 0:
                    return np.Inf
                else:
                    return 0
            else:
                r = -(B*y - D) / w
                if r < 0:
                    return None
                else:
                    return r

    def __neg__(self):
        '''
        magic method to get reverse normal plane
        :return:
        '''
        r = Plane(self.point2, self.point1, self.point0)
        return r

