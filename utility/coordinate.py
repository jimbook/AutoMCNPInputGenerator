
import numpy as np

#just to get a specific class
_tdv = np.array([0.,0.,0.])

#将numpy限定为三维向量
class threeDVector(type(_tdv)):
    def __init__(self, x: float, y: float, z: float):
        super().__init__([x,y,z])

    def homogeneous_vector(self):
        return np.array([*self,1.])

#定义坐标系
class coordinateSystem():
    def __init__(self):
        #默认生成原始坐标系
        self.translationMatrix:np.array = np.eye(4)
        self.rotationMatrix:np.array = np.eye(4)

    @staticmethod
    def translation_matrix(v:threeDVector) -> np.array:
        """

        :param v: 平移向量，x,y,z
        :return: 平移变换矩阵,4x4
        """
        tmp = np.eye(4)
        tmp[3, :3] = v
        return tmp

    @staticmethod
    def X_rotation_matrix(theta:float) -> np.array:
        '''

        :param theta: 弧度，逆时针为正
        :return: 绕X轴的旋转矩阵
        '''
        tmp = np.eye(4)
        tmp[1,1] = np.cos(theta)
        tmp[1,2] = np.sin(theta)
        tmp[2,1] = -np.sin(theta)
        tmp[2,2] = np.cos(theta)
        return tmp

    @staticmethod
    def Y_rotation_matrix(theta:float) -> np.array:
        '''

        :param theta: 弧度，逆时针为正
        :return: 绕Y轴的旋转矩阵
        '''
        tmp = np.eye(4)
        tmp[0, 0] = np.cos(theta)
        tmp[0, 2] = np.sin(theta)
        tmp[2, 0] = -np.sin(theta)
        tmp[2, 2] = np.cos(theta)

    @staticmethod
    def Z_rotation_matrix(theta:float) -> np.array:
        '''

        :param theta: 弧度，逆时针为正
        :return: 绕Y轴的旋转矩阵
        '''
        tmp = np.eye(4)
        tmp[0, 0] = np.cos(theta)
        tmp[0, 1] = np.sin(theta)
        tmp[1, 0] = -np.sin(theta)
        tmp[1, 1] = np.cos(theta)
        return tmp

    @staticmethod
    def rotate_matrix(v:threeDVector, delta: float) -> np.array:
        '''

        :param v: 旋转中心轴
        :param delta: 弧度，逆时针为正
        :return: 绕指定轴旋转的旋转矩阵
        '''
        theta = np.arctan(v[2] / v[1])
        phi = np.arctan(v[1] / (np.sqrt(v[0]*v[0] + v[1]*v[1])))
        D = (coordinateSystem.Y_rotation_matrix(theta)*coordinateSystem.Z_rotation_matrix(phi)*coordinateSystem.
             X_rotation_matrix(delta)*coordinateSystem.Z_rotation_matrix(-phi)*coordinateSystem.Y_rotation_matrix(-theta))
        return D

    def from_origin_to_here(self,point:threeDVector) -> threeDVector:
        '''

        :param point: 原始坐标系下的三维向量
        :return: 当前坐标系下的三维向量
        '''
        t = self.translationMatrix*self.rotationMatrix*point.homogeneous_vector()
        return threeDVector(*t)

    def from_here_to_origin(self,point:threeDVector) -> threeDVector:
        '''

        :param point: 当前坐标系下的三维向量
        :return: 原始坐标系下的三维向量
        '''
        t = np.linalg.inv(self.translationMatrix * self.rotationMatrix * point.homogeneous_vector())
        return threeDVector(*t)

    def move_to(self,v:threeDVector):
        '''
        向对应方向平移
        :param v:
        :return:
        '''
        self.translationMatrix *= self.translation_matrix(v)

    def rotate(self,v:threeDVector,delta:float):
        '''
        绕轴旋转
        :param v:中心轴
        :param delta:弧度，逆时针为正
        :return:
        '''
        self.rotationMatrix *= self.rotate_matrix(v, delta)

class Point(threeDVector):
    def __init__(self, x:float, y:float, z:float, coord: coordinateSystem = None):
        '''

        :param x,y,z: 坐标
        :param coord: 坐标基于的坐标系，如果为None则代表是原始坐标系下的坐标
        '''
        super().__init__(x,y,z)
        self.coord = coord

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
    def get_point_in_coord(cls, point:'Point', coord:coordinateSystem) -> 'Point':
        '''
        获取对应点在对应坐标系下的坐标
        :param point:
        :param coord: 目标坐标系
        :return:
        '''
        if not point.is_in_origin():
            point = point.get_point_in_origin()
        t = coord.from_origin_to_here(point)
        return Point(*t, coord=coord)


if __name__ == '__main__':
    print("Hollow world")



