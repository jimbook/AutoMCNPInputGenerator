import numpy as np

from utility.stratumModel import *
from utility import coordinate, geometry
import unittest
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

outer_diameter = 80
inner_diameter = 10
areaOfCircle = np.pi * (outer_diameter * outer_diameter - inner_diameter * inner_diameter)


def process_function_one_plane_one_boundary_centre_plane(argTulpe: tuple[float, float]):
    y = argTulpe[0]
    theta = argTulpe[1]
    dZone = DetectionZone(coordinate.CoordinateSystem(), outer_diameter, inner_diameter)
    coord = coordinate.CoordinateSystem().rotate(coordinate.ThreeDVector(0, 1, 0), theta)
    p1 = coordinate.Point(outer_diameter, 0, 0, coord).get_point_in_origin()
    p2 = coordinate.Point(-outer_diameter, 0, 0, coord).get_point_in_origin()
    p3 = coordinate.Point(0, y, outer_diameter, coord).get_point_in_origin()
    plane_ij = geometry.Plane(p1, p2, p3)

    boundary_ij = OnePlaneBoundary(plane_ij, dZone)
    s_ij = Stratum(boundary_ij, 1., dZone)
    return s_ij.volume

def process_function_one_plane_one_boundary_move_plane(argTulpe: tuple[float, float]):
    y = argTulpe[0]
    theta = argTulpe[1]
    v_stand = (outer_diameter - y) * areaOfCircle

    d = min(outer_diameter - y, y + outer_diameter)
    if d > 3:
        phi_y_i = np.linspace(y - d, y + d, 30)
    else:
        phi_y_i = np.arange(y - d, y + d, 0.1)
    dZone = DetectionZone(coordinate.CoordinateSystem(), outer_diameter, inner_diameter)
    coord = coordinate.CoordinateSystem().rotate(coordinate.ThreeDVector(0, 1, 0), theta)

    r = []
    for i in range(phi_y_i.shape[0]):
        phi_y = phi_y_i[i]
        p1 = coordinate.Point(outer_diameter, y, 0, coord).get_point_in_origin()
        p2 = coordinate.Point(-outer_diameter, y, 0, coord).get_point_in_origin()
        p3 = coordinate.Point(0, phi_y, outer_diameter, coord).get_point_in_origin()
        plane_ij = geometry.Plane(p1, p2, p3)

        boundary_ij = OnePlaneBoundary(plane_ij, dZone)
        s_ij = Stratum(boundary_ij, 1., dZone)
        r.append(s_ij.volume)
    if len(r) == 0:
        return 0.
    r_np = np.array(r)
    r0 = r[0]
    if not np.allclose(r_np, v_stand, rtol=5.0e-3 * v_stand):
        return -1.
    else:
        return r0

def process_function_one_plane_two_boundary_move_plane(argTulpe:tuple[float, float]):
    y_1 = argTulpe[0]
    phi_1 = argTulpe[1]
    phi_j = np.linspace(0, 360, 24)
    d = min(outer_diameter - y_1, outer_diameter + y_1)
    diff_y_i = np.arange(y_1, outer_diameter - d, 1.)
    if d > 3:
        shift_k = np.linspace(-d, d, 10)
    else:
        shift_k = np.arange(-d, d, 0.6)
    for i in range(diff_y_i.shape[0]):
        y_2 = diff_y_i[i]
        for k in range(shift_k.shape[0]):
            shiff_y_1 = shift_k[k]
            for k_ in range(shift_k.shape[0]):
                shiff_y_2 =shift_k[k_]
                for j in range(phi_j.shape[0]):
                    phi_2 = phi_j[j]
                    dZone = DetectionZone(coordinate.CoordinateSystem(), outer_diameter, inner_diameter)
                    coord_1 = coordinate.CoordinateSystem().rotate(coordinate.ThreeDVector(0, 1, 0), phi_1)
                    p1_1 = coordinate.Point(outer_diameter, y_1, 0, coord_1).get_point_in_origin()
                    p2_1 = coordinate.Point(-outer_diameter, y_1, 0, coord_1).get_point_in_origin()
                    p3_1 = coordinate.Point(0, shiff_y_1 + y_1, outer_diameter, coord_1).get_point_in_origin()
                    plane_1 = geometry.Plane(p1_1, p2_1, p3_1)
                    boundary_1 = OnePlaneBoundary(plane_1, dZone)

                    coord_2 = coordinate.CoordinateSystem().rotate(coordinate.ThreeDVector(0, 1, 0), phi_2)
                    p1_2 = coordinate.Point(outer_diameter, y_2, 0, coord_2).get_point_in_origin()
                    p2_2 = coordinate.Point(-outer_diameter, y_2, 0, coord_2).get_point_in_origin()
                    p3_2 = coordinate.Point(0, shiff_y_2 + y_2, outer_diameter, coord_2).get_point_in_origin()
                    plane_2 = geometry.Plane(p3_2, p2_2, p1_2)
                    boundary_2 = OnePlaneBoundary(plane_2, dZone)

                    # check if two plane is intersected
                    check_p1_1 = p3_1
                    check_p1_2 = coordinate.Point(0, -shiff_y_1 + y_1, -outer_diameter, coord_1).get_point_in_origin()
                    if plane_2.isUnderIt(check_p1_1) or plane_2.isUnderIt(check_p1_2):
                        continue


                    boundary = UnionAndPlanesBoundary(boundary_1, boundary_2)
                    s = Stratum(boundary, 1., dZone)
                    if s.volume != areaOfCircle * abs(y_1 - y_2):
                        return -1
        return 1

class StratumVolumeTestCase(unittest.TestCase):
    def _test_central_vertical_plane(self):
        # 1. 构建一个下界面是xOz平面的地层，下界面法向量朝向x正方向
        plane_0 = geometry.Plane(coordinate.Point(outer_diameter, 0, 0),
                                        coordinate.Point(-outer_diameter,0,  0),
                                        coordinate.Point(-outer_diameter,0,  -1))
        print("normal:", plane_0.normal)
        downBoundary_0 = OnePlaneBoundary(plane_0)
        s_0 = Stratum(downBoundary_0, 1., None)
        # 通过体积公式计算体积
        v_0 = outer_diameter * areaOfCircle
        self.assertAlmostEqual(s_0.volume, v_0)


    def _test_onePlane_oneBoundary(self):
        y_i = np.linspace(-outer_diameter, outer_diameter, 80)
        theta_j = np.linspace(0, 360, 24)

        processArg = [(i, j) for i in y_i for j in theta_j]
        with Pool(cpu_count()) as p:
            result = p.imap(process_function_one_plane_one_boundary_centre_plane, processArg)
            idx_v = 0
            errorList = []
            for r in tqdm(result, total=len(processArg)):
                if r == -1:
                    errorList.append(processArg[idx_v])
                idx_v += 1
            self.assertEqual(len(errorList), 0,
                                msg="\nerror in these argument:\n{} \n {}".format(errorList, '=' * 20))

    def test_onePlane_move(self):
        y_i = np.linspace(-outer_diameter, outer_diameter, 80)
        theta_j = np.linspace(0, 360, 24)
        processArg = [(i, j) for i in y_i for j in theta_j]
        with Pool(cpu_count()) as p:
            result = p.imap(process_function_one_plane_one_boundary_move_plane, processArg)
            idx_v = 0
            errorList = []
            for r in tqdm(result, total=len(processArg)):
                if r == -1:
                    errorList.append(processArg[idx_v])
                idx_v += 1
            self.assertEqual(len(errorList), 0,
                                msg="\nerror in these argument:\n{} \n {}".format(errorList, '=' * 20))

    def test_onePlane_twoBoundary(self):
        pass

    def test_twoPlane_oneBoundary(self):
        pass

    def test_twoPlane_twoBoundary(self):
        pass

if __name__ == "main":
    unittest.main()