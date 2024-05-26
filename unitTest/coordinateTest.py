import numpy as np

from utility.coordinate import *
import unittest
class TestCoordinate(unittest.TestCase):
    def assertEqualVector(self, first, second):
        return self.assertTrue(np.allclose(first, second), msg="\ntest result is: \n{}"
                                                        "\nexpected result is: \n{}".format(first, second))

    def test_3DVector(self):
        t = threeDVector(float(1.),float(2.),float(3.))
        self.assertEqualVector(t.homogeneous_vector(), np.array([1.,2.,3.,1.]))

    def test_TranslationMatrix(self):
        moveVector = threeDVector(float(1.),float(2.),float(3.))
        answer = np.array([[1.,0.,0.,-1.],
                           [0.,1.,0.,-2.],
                           [0.,0.,1.,-3.],
                           [0.,0.,0.,1.]])
        tm = coordinateSystem.translation_matrix(moveVector)
        self.assertEqualVector(tm, answer)
        p = Point(1.,2.,3.)
        p_answer = np.dot(coordinateSystem.translation_matrix(p), p.homogeneous_vector())
        self.assertEqualVector(p_answer, np.array([0,0,0,1]))

    def test_RotationMatrix(self):
        # 绕XYZ轴旋转矩阵
        p_x = Point(1., 0., 0.)
        p_y = Point(0, 1, 0)
        p_z = Point(0, 0, 1.)
        ## 90度旋转
        x_90 = coordinateSystem.X_rotation_matrix(90)
        y_90 = coordinateSystem.Y_rotation_matrix(90)
        z_90 = coordinateSystem.Z_rotation_matrix(90)
        answer = [
            # x_90
            np.array([1, 0, 0 ,1]),
            np.array([0, 0, -1, 1]),
            np.array([0, 1, 0, 1]),
            # y_90
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
            np.array([-1, 0, 0, 1]),
            # z_90
            np.array([0, -1, 0, 1]),
            np.array([1, 0, 0, 1]),
            np.array([0, 0, 1, 1])
        ]
        answerIdx = 0
        for rotation in [x_90, y_90, z_90]:
            for point in [p_x, p_y, p_z]:
                p_result = np.dot(rotation,point.homogeneous_vector())
                self.assertEqualVector(p_result, answer[answerIdx])
                answerIdx += 1
        ## 180度旋转
        x_180 = coordinateSystem.X_rotation_matrix(180)
        y_180 = coordinateSystem.Y_rotation_matrix(180)
        z_180 = coordinateSystem.Z_rotation_matrix(180)
        answer = [
            # x_180
            np.array([1, 0, 0, 1]),
            np.array([0, -1, 0, 1]),
            np.array([0, 0, -1, 1]),
            # y_180
            np.array([-1, 0, 0, 1]),
            np.array([0, 1, 0, 1]),
            np.array([0, 0, -1, 1]),
            # z_180
            np.array([-1, 0, 0, 1]),
            np.array([0, -1, 0, 1]),
            np.array([0, 0, 1, 1])
        ]
        answerIdx = 0
        for rotation in [x_180, y_180, z_180]:
            for point in [p_x, p_y, p_z]:
                p_result = np.dot(rotation, point.homogeneous_vector())
                self.assertEqualVector(p_result, answer[answerIdx])
                answerIdx += 1
        ##45度旋转
        x_45 = coordinateSystem.X_rotation_matrix(45)
        y_45 = coordinateSystem.Y_rotation_matrix(45)
        z_45 = coordinateSystem.Z_rotation_matrix(45)
        answer = [
            # x_45
            np.array([1, 0, 0, 1]),
            np.array([0, np.sqrt(2)/2, -np.sqrt(2)/2, 1]),
            np.array([0, np.sqrt(2)/2, np.sqrt(2)/2, 1]),
            # y_45
            np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2, 1]),
            np.array([0, 1, 0, 1]),
            np.array([-np.sqrt(2)/2, 0, np.sqrt(2)/2, 1]),
            # z_45
            np.array([np.sqrt(2)/2, -np.sqrt(2)/2, 0, 1]),
            np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 1]),
            np.array([0, 0, 1, 1])
        ]
        answerIdx = 0
        for rotation in [x_45, y_45, z_45]:
            for point in [p_x, p_y, p_z]:
                p_result = np.dot(rotation, point.homogeneous_vector())
                self.assertEqualVector(p_result, answer[answerIdx])
                answerIdx += 1

        #绕指定轴旋转
        # 指定x轴
        axis_x = threeDVector(1, 0, 0)
        axis_y = threeDVector(0, 1, 0)
        axis_z = threeDVector(0, 0, 1)
        ## 旋转90度
        x_90 = coordinateSystem.rotate_matrix(axis_x, 90)
        y_90 = coordinateSystem.rotate_matrix(axis_y, 90)
        z_90 = coordinateSystem.rotate_matrix(axis_z, 90)
        answer = [
            # x_90
            np.array([1, 0, 0, 1]),
            np.array([0, 0, -1, 1]),
            np.array([0, 1, 0, 1]),
            # y_90
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
            np.array([-1, 0, 0, 1]),
            # z_90
            np.array([0, -1, 0, 1]),
            np.array([1, 0, 0, 1]),
            np.array([0, 0, 1, 1])
        ]
        answerIdx = 0
        for rotation in [x_90, y_90, z_90]:
            for point in [p_x, p_y, p_z]:
                p_result = np.dot(rotation, point.homogeneous_vector())
                self.assertEqualVector(p_result, answer[answerIdx])
                answerIdx += 1



        #绕指定轴（1，1，1）旋转
        ## 旋转120，正好能落到轴上
        v = threeDVector(1.,1.,1.)
        v_120 = coordinateSystem.rotate_matrix(v, 120)
        answer = [
            np.array([0,0,1,1]), # p_x will locate at y axis
            np.array([1,0,0,1]), # p_y will locate at z axis
            np.array([0,1,0,1]), # p_z will locate at x axis
        ]
        answerIdx = 0
        for point in [p_x, p_y, p_z]:
            p_result = np.dot(v_120, point.homogeneous_vector())
            self.assertEqualVector(p_result, answer[answerIdx])
            answerIdx += 1

    def test_vectorTransformFunction(self):
        pointList:list[threeDVector] = [
            threeDVector(1, 0, 0),
            threeDVector(0, 1, 0),
            threeDVector(0, 0, 1),
            threeDVector(1, 1, 1),
            threeDVector(-1, -1, 1),
            threeDVector(1,2,3)
        ]

        # 测试平移坐标系+向量变换
        c_0 = coordinateSystem().move_to(pointList[0])
        answer = list(map(lambda p: p - np.array([1, 0, 0]), pointList))
        answerIdx = 0
        for v in pointList:
            after_p = c_0.from_origin_to_here(v)
            back_p = c_0.from_here_to_origin(after_p)
            self.assertEqualVector(after_p, answer[answerIdx])
            self.assertEqualVector(back_p, v)
            answerIdx += 1

        c_1 = c_0.move_to(np.array([0, -2, 0]))
        answer = list(map(lambda p: p - np.array([0, -2, 0]), answer))
        answerIdx = 0
        for v in pointList:
            after_p = c_1.from_origin_to_here(v)
            back_p = c_1.from_here_to_origin(after_p)
            self.assertEqualVector(after_p, answer[answerIdx])
            self.assertEqualVector(back_p, v)
            answerIdx += 1

        c_2 = c_1.move_to(np.array([0, 0, 3]))
        answer = list(map(lambda p: p - np.array([1, -2, 3]), pointList))
        answerIdx = 0
        for v in pointList:
            after_p = c_2.from_origin_to_here(v)
            back_p = c_2.from_here_to_origin(after_p)
            self.assertEqualVector(after_p, answer[answerIdx])
            self.assertEqualVector(back_p, v)
            answerIdx += 1

        # 测试旋转坐标系+向量变换
        ## 基于坐标轴旋转
        ### X轴
        c_3 = coordinateSystem().rotate(threeDVector(1., 0, 0), 90)
        answer = [
            threeDVector(1, 0, 0),
            threeDVector(0,0,-1),
            threeDVector(0,1,0),
            threeDVector(1,1,-1),
            threeDVector(-1,1,1),
            threeDVector(1, 3, -2)
        ]
        answerIdx = 0
        for v in pointList:
            after_p = c_3.from_origin_to_here(v)
            back_p = c_3.from_here_to_origin(after_p)
            self.assertEqualVector(after_p, answer[answerIdx])
            self.assertEqualVector(back_p, v)
            answerIdx += 1

        ### Y轴
        c_4 = coordinateSystem().rotate(threeDVector(0, 1, 0), 90)
        answer = [
            threeDVector(0, 0, 1),
            threeDVector(0, 1, 0),
            threeDVector(-1, 0, 0),
            threeDVector(-1, 1, 1),
            threeDVector(-1, -1, -1),
            threeDVector(-3, 2, 1)
        ]
        answerIdx = 0
        for v in pointList:
            after_p = c_4.from_origin_to_here(v)
            back_p = c_4.from_here_to_origin(after_p)
            self.assertEqualVector(after_p, answer[answerIdx])
            self.assertEqualVector(back_p, v)
            answerIdx += 1

        ### Z轴
        c_5 = coordinateSystem().rotate(threeDVector(0, 0, 1), 90)
        answer = [
            threeDVector(0, -1, 0),
            threeDVector(1, 0, 0),
            threeDVector(0, 0, 1),
            threeDVector(1, -1, 1),
            threeDVector(-1, 1, 1),
            threeDVector(2, -1, 3)
        ]
        answerIdx = 0
        for v in pointList:
            after_p = c_5.from_origin_to_here(v)
            back_p = c_5.from_here_to_origin(after_p)
            self.assertEqualVector(after_p, answer[answerIdx])
            self.assertEqualVector(back_p, v)
            answerIdx += 1
        ## 基于特定轴(1, 1, 1)旋转, x->z, y->x, z->y
        c_6 = coordinateSystem().rotate(threeDVector(1, 1, 1), 120)
        answer = [
            threeDVector(0, 0, 1),
            threeDVector(1, 0, 0),
            threeDVector(0, 1, 0),
            threeDVector(1, 1, 1),
            threeDVector(-1, 1, -1),
            threeDVector(2, 3, 1)
        ]
        answerIdx = 0
        for v in pointList:
            after_p = c_6.from_origin_to_here(v)
            back_p = c_6.from_here_to_origin(after_p)
            self.assertEqualVector(after_p, answer[answerIdx])
            self.assertEqualVector(back_p, v)
            answerIdx += 1

    def test_pointInCoordinateSystem(self):
        pointList_original = [
            Point(0, 0, 0),
            Point(1, 0, 0),
            Point(0, 1, 0),
            Point(0, 0, 1),
            Point(1, 1, 1),
            Point(1, 2, 3)
        ]
        # 平移变换后的坐标系
        O_c_0 = Point(1, 1, 1)
        c_0 = coordinateSystem().move_to(O_c_0)
        answer = list(map(lambda p: p - O_c_0, pointList_original))
        answerIdx = 0
        for p in pointList_original:
            p_c_0 = Point.get_point_in_coord(p, c_0)
            self.assertEqualVector(p_c_0, answer[answerIdx])
            answerIdx += 1

        O_c_1 = Point(5, -2, 3)
        c_1 = coordinateSystem().move_to(O_c_1)
        answer = list(map(lambda p: p - O_c_1, pointList_original))
        answerIdx = 0
        for v in pointList_original:
            p_c_1 = Point.get_point_in_coord(v, c_1)
            self.assertEqualVector(p_c_1, answer[answerIdx])
            answerIdx += 1

        #旋转变换后的坐标系
        ## 绕轴旋转90度
        axis_x = threeDVector(1, 0, 0)
        axis_y = threeDVector(0, 1, 0)
        axis_z = threeDVector(0, 0, 1)
        c_x90 = coordinateSystem().rotate(axis_x, 90)
        c_y90 = coordinateSystem().rotate(axis_y, 90)
        c_z90 = coordinateSystem().rotate(axis_z, 90)
        answer_x90 = [
            threeDVector(0, 0, 0),
            threeDVector(1, 0, 0),
            threeDVector(0, 0, -1),
            threeDVector(0, 1, 0),
            threeDVector(1, 1, -1),
            threeDVector(1, 3, -2)
        ]
        answer_y90 = [
            threeDVector(0, 0, 0),
            threeDVector(0, 0, 1),
            threeDVector(0, 1, 0),
            threeDVector(-1, 0, 0),
            threeDVector(-1, 1, 1),
            threeDVector(-3, 2, 1)
        ]
        answer_z90 = [
            threeDVector(0, 0, 0),
            threeDVector(0, -1, 0),
            threeDVector(1, 0, 0),
            threeDVector(0, 0, 1),
            threeDVector(1, -1, 1),
            threeDVector(2, -1, 3)
        ]
        answerIdx = 0
        for p in pointList_original:
            p_c_x90 = Point.get_point_in_coord(p, c_x90)
            p_c_y90 = Point.get_point_in_coord(p, c_y90)
            p_c_z90 = Point.get_point_in_coord(p, c_z90)
            self.assertEqualVector(p_c_x90, answer_x90[answerIdx])
            self.assertEqualVector(p_c_y90, answer_y90[answerIdx])
            self.assertEqualVector(p_c_z90, answer_z90[answerIdx])
            answerIdx += 1

        ## 绕轴旋转45度
        c_x45 = coordinateSystem().rotate(axis_x, 45)
        c_y45 = coordinateSystem().rotate(axis_y, 45)
        c_z45 = coordinateSystem().rotate(axis_z, 45)
        answer_x45 = [
            threeDVector(0, 0, 0),
            threeDVector(1, 0, 0),
            threeDVector(0, np.sqrt(2) / 2, -np.sqrt(2) / 2),
            threeDVector(0, np.sqrt(2) / 2, np.sqrt(2) / 2),
            threeDVector(1, np.sqrt(2), 0)
        ]
        answer_y45 = [
            threeDVector(0, 0, 0),
            threeDVector(np.sqrt(2) / 2, 0, np.sqrt(2) / 2),
            threeDVector(0, 1, 0),
            threeDVector(-np.sqrt(2) / 2, 0, np.sqrt(2) / 2),
            threeDVector(0, 1, np.sqrt(2))
        ]
        answer_z45 = [
            threeDVector(0, 0, 0),
            threeDVector(np.sqrt(2) / 2, -np.sqrt(2) / 2, 0),
            threeDVector(np.sqrt(2) / 2, np.sqrt(2) / 2, 0),
            threeDVector(0, 0, 1),
            threeDVector(np.sqrt(2), 0, 1)
        ]
        answerIdx = 0
        for p in pointList_original[:-1]:
            p_c_x45 = Point.get_point_in_coord(p, c_x45)
            p_c_y45 = Point.get_point_in_coord(p, c_y45)
            p_c_z45 = Point.get_point_in_coord(p, c_z45)
            print(answerIdx)
            self.assertEqualVector(p_c_x45, answer_x45[answerIdx])
            self.assertEqualVector(p_c_y45, answer_y45[answerIdx])
            self.assertEqualVector(p_c_z45, answer_z45[answerIdx])
            answerIdx += 1

        ## 绕(1, 1, 1)旋转120度
        c_v120 = coordinateSystem().rotate(threeDVector(1, 1, 1),120)
        answer = [
            threeDVector(0, 0, 0),
            threeDVector(0, 0, 1),
            threeDVector(1, 0, 0),
            threeDVector(0, 1, 0),
            threeDVector(1, 1, 1),
            threeDVector(2, 3, 1)
        ]
        answerIdx = 0
        for v in pointList_original:
            p_c_v1 = Point.get_point_in_coord(v, c_v120)
            self.assertEqualVector(p_c_v1, answer[answerIdx])
            answerIdx += 1

        # 平移加旋转变换
        ## 平移后绕轴旋转
        c_2 = coordinateSystem().move_to(threeDVector(2,2,2))
        c_2_x90 = c_2.rotate(axis_x, 90)
        answer_x90 = [
            threeDVector(-2, -2, 2),
            threeDVector(-1, -2, 2),
            threeDVector(-2, -2, 1),
            threeDVector(-2, -1, 2),
            threeDVector(-1, -1, 1),
            threeDVector(-1, 1, 0)
        ]
        answerIdx = 0
        for v in pointList_original:
            p_c_2x90 = Point.get_point_in_coord(v, c_2_x90)
            self.assertEqualVector(p_c_2x90, answer_x90[answerIdx])
            answerIdx += 1
        print("end")


if __name__ == '__main__':
    unittest.main()
