from typing import Any

import numpy as np
from copy import deepcopy

#just to get a specific class
_tdv = np.array([0.,0.,0.])

#将numpy限定为三维向量
class ThreeDVector(np.ndarray):
    def __new__(cls, x: float, y: float, z: float):
        obj = np.asarray([x, y, z]).view(cls)
        return obj

    def __init__(self, a,b,c):
        pass

    def homogeneous_vector(self) -> np.ndarray:
        return np.append(self, 1.)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

#一个辅助函数，角度转弧度
def angle_to_radian(angle: float) -> float:
    return angle * np.pi / 180.

#一个辅助函数，弧度转角度
def radian_to_angle(radian: float) -> float:
    return radian * 180. / np.pi

#定义坐标系
class CoordinateSystem():
    def __init__(self):
        #默认生成原始坐标系，默认为右手系
        self.translationMatrix:np.array = np.eye(4)
        self.rotationMatrix:np.array = np.eye(4)
        self.right_handed = True

    @staticmethod
    def translation_matrix(v:ThreeDVector) -> np.array:
        """

        :param v: 平移向量，x,y,z
        :return: 平移变换矩阵,4x4
        """
        tmp = np.eye(4)
        tmp[:3, 3] = -v
        return tmp

    @staticmethod
    def X_rotation_matrix(theta:float) -> np.array:
        '''

        :param theta: 角度，逆时针为正
        :return: 绕X轴的旋转矩阵
        '''
        theta = angle_to_radian(theta)
        tmp = np.eye(4)
        tmp[1,1] = np.cos(theta)
        tmp[1,2] = np.sin(theta)
        tmp[2,1] = -np.sin(theta)
        tmp[2,2] = np.cos(theta)
        return tmp

    @staticmethod
    def Y_rotation_matrix(theta:float) -> np.array:
        '''

        :param theta: 角度，逆时针为正
        :return: 绕Y轴的旋转矩阵
        '''
        theta = angle_to_radian(theta)
        tmp = np.eye(4)
        tmp[0, 0] = np.cos(theta)
        tmp[0, 2] = -np.sin(theta)
        tmp[2, 0] = np.sin(theta)
        tmp[2, 2] = np.cos(theta)
        return tmp

    @staticmethod
    def Z_rotation_matrix(theta:float) -> np.array:
        '''

        :param theta: 角度，逆时针为正
        :return: 绕Y轴的旋转矩阵
        '''
        theta = angle_to_radian(theta)
        tmp = np.eye(4)
        tmp[0, 0] = np.cos(theta)
        tmp[0, 1] = np.sin(theta)
        tmp[1, 0] = -np.sin(theta)
        tmp[1, 1] = np.cos(theta)
        return tmp

    @staticmethod
    def rotate_matrix(v:ThreeDVector, delta: float) -> np.array:
        '''

        :param v: 旋转中心轴
        :param delta: 角度，逆时针为正
        :return: 绕指定轴旋转的旋转矩阵
        '''
        if v[0] == 0:
            theta = 90
        else:
            theta = np.arctan(v[1] / v[0]) / np.pi * 180
        if v[2] == 0:
            phi = 90
        else:
            phi = np.arctan((np.sqrt(v[0]*v[0] + v[1]*v[1]) / v[2])) / np.pi * 180
        matrixD = CoordinateSystem.Z_rotation_matrix(theta)
        matrixD = np.dot(CoordinateSystem.Y_rotation_matrix(phi), matrixD)
        matrixD = np.dot(CoordinateSystem.Z_rotation_matrix(delta), matrixD)
        matrixD = np.dot(CoordinateSystem.Y_rotation_matrix(-phi), matrixD)
        matrixD = np.dot(CoordinateSystem.Z_rotation_matrix(-theta), matrixD)
        return matrixD

    @property
    def transformMatrix(self):
        return np.dot(self.rotationMatrix, self.translationMatrix)

    @property
    def inverseTransformMatrix(self):
        return np.linalg.inv(self.transformMatrix)

    def from_origin_to_here(self, vector:ThreeDVector) -> ThreeDVector:
        '''

        :param vector: 原始坐标系下的三维向量
        :return: 当前坐标系下的三维向量
        '''
        tmp_v = np.dot(self.transformMatrix, vector.homogeneous_vector())
        if not self.right_handed:
            tmp_v[2] = -tmp_v[2]
        return ThreeDVector(*tmp_v[:3])

    def from_here_to_origin(self, vector:ThreeDVector) -> ThreeDVector:
        '''

        :param vector: 当前坐标系下的三维向量
        :return: 原始坐标系下的三维向量
        '''
        if not self.right_handed:
            vector[2] = -vector[2]
        tmp_v = np.dot(self.inverseTransformMatrix, vector.homogeneous_vector())
        return ThreeDVector(*tmp_v[:3])
    #
    # def get_point_in_here(self, point:'Point') -> 'Point':
    #     if point.coord is None:
    #         new_vector = self.from_here_to_origin(point)
    #     else:
    #         original_vector = point.coord.from_here_to_origin(point)
    #         new_vector = self.from_here_to_origin(original_vector)
    #     return Point(*new_vector, coord=self)

    def move_to(self,v:ThreeDVector):
        '''
        向对应方向平移
        :param v:
        :return:
        '''
        new_coordinate = deepcopy(self)
        new_coordinate.translationMatrix = np.dot(self.translation_matrix(v),new_coordinate.translationMatrix)
        return new_coordinate

    def rotate(self,v:ThreeDVector,delta:float):
        '''
        绕轴旋转
        :param v:中心轴
        :param delta:角度，逆时针为正
        :return:
        '''
        new_coordinate = deepcopy(self)
        new_coordinate.rotationMatrix = np.dot(new_coordinate.rotate_matrix(v, delta), new_coordinate.rotationMatrix)
        return new_coordinate

    def switchToOtherHandedSystem(self, right_handed:bool = True):
        new_coordinate = deepcopy(self)
        new_coordinate.right_handed = right_handed
        return new_coordinate

    def __eq__(self, other):
        if other is None:
            other = CoordinateSystem()
        if np.array_equal(self.translationMatrix, other.translationMatrix) and \
                np.array_equal(self.rotationMatrix, other.rotationMatrix):
            return True
        else:
            return False



# class CylindricalCoordinateSystem(CoordinateSystem):
#     def from_here_to_origin(self, vector:ThreeDVector) -> ThreeDVector:
#         v = super().from_here_to_origin(vector)
#


class Point(ThreeDVector):
    def __new__(cls, x:float, y:float, z:float, coord: CoordinateSystem = None):
        '''
        :param x,y,z: 坐标
        :param coord: 坐标基于的坐标系，如果为None则代表是原始坐标系下的坐标
        '''
        j = super().__new__(cls,x,y,z)
        j.coord = coord
        return j

    def __init__(self, x:float, y:float, z:float, coord: CoordinateSystem = None):
        self.coord = coord

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.coord = getattr(obj, 'info', None)

    def is_in_origin(self) -> bool:
        '''
        自身的坐标是否是基于原始坐标系
        :return: true-基于原始坐标系; false-基于其他坐标系
        '''
        if self.coord is None:
            return True
        else:
            return False

    def get_point_in_origin(self) -> 'Point':
        '''
        获取基于原始坐标系下的坐标
        :return:
        '''
        if self.is_in_origin():
            return self
        else:
            t = self.coord.from_here_to_origin(self)
            return Point(*t)

    @classmethod
    def get_point_in_coord(cls, point:'Point', coord:CoordinateSystem) -> 'Point':
        '''
        获取对应点在对应坐标系下的坐标
        :param point:
        :param coord: 目标坐标系
        :return:
        '''
        if not point.is_in_origin():
            point = point.get_point_in_origin()
        if coord is None:
            t = point
        else:
            t = coord.from_origin_to_here(point)
        return Point(*t, coord=coord)

    def get_in_coord(self, coord:CoordinateSystem) -> 'Point':
        return Point.get_point_in_coord(self,coord)

def point_equal(pointA:Point, pointB:Point) -> bool:
    if pointA.is_in_origin():
        pA = pointA
    else:
        pA = pointA.get_point_in_origin()
    if pointB.is_in_origin():
        pB = pointB
    else:
        pB = pointB.get_point_in_origin()
    return np.allclose(pA, pB)


if __name__ == '__main__':
    print(Point(1.,1.,1.))
    print(CoordinateSystem.X_rotation_matrix(30/180. * np.pi))



