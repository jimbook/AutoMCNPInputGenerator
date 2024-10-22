import numpy as np
import pandas as pd

from MCNPInput import MCNPAutoInput, GeometricModel, Material, SourceDefine,AuxiliaryFunction
from MCNPInput.MCNPAutoInput import MCNP_CardsPool
from utility.stratumModel import SingleStratumModel, DetectionZone
from utility import coordinate, geometry, stratumModel
from copy import deepcopy




class BaseDetector(MCNPAutoInput.CellsPackage):
    def create_cells(self):
        Material.init_a_inner_material_database()
        #钻井液导流通道
        surf_up_0 = GeometricModel.MCNP_PlaneSurface('py',[40],note='Drill collar start plane position', transformation=self.transform)
        surf_down_0 = GeometricModel.MCNP_PlaneSurface('py', [-40], note='Drill collar stop plane position', transformation=self.transform)
        surf_outer_0 = GeometricModel.MCNP_surface('cy', [2.5], note='mud pipeline', transformation=self.transform)

        cell0 = GeometricModel.MCNP_cell(Material.MaterialStorage().get_material_from_name("H2O"), 1, note="Drilling fluid diversion channel")
        cell0.addSurface(surf_up_0, -1)
        cell0.addSurface(surf_outer_0, -1)
        cell0.addSurface(surf_down_0)
        self.cells.append(cell0)


        # 探测器
        surf_up_1 = GeometricModel.MCNP_PlaneSurface('py', [15], note='Detector start plane position', transformation=self.transform)
        surf_down_1 = GeometricModel.MCNP_PlaneSurface('py', [-15], note='Detector stop plane position', transformation=self.transform)
        surf_fill_outer_0 = GeometricModel.MCNP_surface('cy', [8], note='Detector edge outer',
                                                        transformation=self.transform)
        surf_inner_3 = GeometricModel.MCNP_surface('cy', [8], note='inner of Sealing substance',
                                                   transformation=self.transform)
        surf_outer_3 = GeometricModel.MCNP_surface('cy', [9], note='outer of drill collar',
                                                   transformation=self.transform)

        cell_drillCollar = GeometricModel.MCNP_cell(Material.MaterialStorage().get_material_from_name("W-Ni-Fe"), 7.86,
                                                    note='drill collar')
        cell_drillCollar.addSurface(surf_up_0, -1)
        cell_drillCollar.addSurface(surf_down_0)
        cell_drillCollar.addSurface(surf_outer_3, -1)
        cell_drillCollar.addSurface(surf_outer_0)

        for i in [-5, 5]:
            for x, y in [(0, i), (i, 0)]:
                surf_outer_1 = GeometricModel.MCNP_surface('c/y', [x, y, 1.5], note='outer of NaI', transformation=self.transform)

                cell1 = GeometricModel.MCNP_cell(Material.MaterialStorage().get_material_from_name("NaI"), 3.76, note="NaI", detector=True)
                cell1.addSurface(surf_up_1, -1)
                cell1.addSurface(surf_down_1)
                cell1.addSurface(surf_outer_1, -1)
                self.cells.append(cell1)

                surf_outer_2 = GeometricModel.MCNP_surface('c/y', [x, y, 2], note='rubber fill outer', transformation=self.transform)
                cell2 = GeometricModel.MCNP_cell(Material.MaterialStorage().get_material_from_name('rubber'), 1.35, note='rubber fill')
                cell2.addSurface(surf_outer_2, -1)
                cell2.addSurface(surf_down_1)
                cell2.addSurface(surf_up_1, -1)
                cell2.addSurface(surf_outer_1)
                self.cells.append(cell2)

                surf_fill_left_0 = GeometricModel.MCNP_PlaneSurface('px' if x == 0 else 'pz', [-2], note='Detector slotted edge negative', transformation=self.transform)
                surf_fill_right_0 = GeometricModel.MCNP_PlaneSurface('px'if x == 0 else 'pz', [2], note='Detector slotted edge positive', transformation=self.transform)
                surf_fill_inner_0 = GeometricModel.MCNP_PlaneSurface('pz' if x == 0 else 'px', [i], note='Detector slotted edge inner',
                                                                     transformation=self.transform)

                cell3 = GeometricModel.MCNP_cell(Material.MaterialStorage().get_material_from_name('epoxy_resin'), 0.98, note='detector slotted')
                cell3.addSurface(surf_fill_left_0)
                cell3.addSurface(surf_fill_right_0, -1)
                cell3.addSurface(surf_fill_outer_0, -1)
                cell3.addSurface(surf_fill_inner_0, int(i/abs(i)))
                cell3.addSurface(surf_outer_2)
                cell3.addSurface(surf_up_1, -1)
                cell3.addSurface(surf_down_1)
                self.cells.append(cell3)


                cell4 = GeometricModel.MCNP_cell(Material.MaterialStorage().get_material_from_name('Be-Cu'), 8.3, note="Be-Cu")
                cell4.addSurface(surf_inner_3)
                cell4.addSurface(surf_outer_3, -1)
                cell4.addSurface(surf_fill_left_0)
                cell4.addSurface(surf_fill_right_0, -1)
                cell4.addSurface(surf_fill_inner_0, int(i/abs(i)))
                cell4.addSurface(surf_up_1, -1)
                cell4.addSurface(surf_down_1)
                self.cells.append(cell4)



                excludeSurface0 = GeometricModel.MCNP_UnionSurface((surf_fill_left_0, -1),
                                                                  (surf_fill_right_0, 1),
                                                                  (surf_fill_inner_0, -int(i/abs(i))),
                                                                  (surf_up_1, 1),
                                                                  (surf_down_1, -1))
                excludeSurface1 = GeometricModel.MCNP_UnionSurface((surf_outer_2, 1),
                                                                   (surf_up_1, 1),
                                                                   (surf_down_1, -1))
                cell_drillCollar.addSurface(excludeSurface0)
                cell_drillCollar.addSurface(excludeSurface1)

        self.cells.append(cell_drillCollar)

        exSurface = GeometricModel.MCNP_UnionSurface((surf_up_0, 1), (surf_down_0, -1), (surf_outer_3, 1))


        self._surface_exclude.append(exSurface)

class Borehole(MCNPAutoInput.CellsPackage):
    def __init__(self, Diameter:float, investigationDepth:float,parent:MCNPAutoInput.CellsPackage = None,transform:GeometricModel.MCNP_transformation = None):
        self.diameter = Diameter
        self.investigationDepth = investigationDepth
        super().__init__(parent=parent, transform=transform)

    def create_cells(self):
        m = Material.MaterialStorage().get_material_from_name("H2O")
        cell = GeometricModel.MCNP_cell(m, 1.0, note='Borehole')

        surf_up_0 = GeometricModel.MCNP_PlaneSurface('py', [self.investigationDepth], 'Upper hole interface')
        surf_down_0 = GeometricModel.MCNP_PlaneSurface('py', [-self.investigationDepth], 'lower hold interface')
        surf_outer_0 = GeometricModel.MCNP_surface('cy', [self.diameter], 'Borehole boundary')
        cell.addSurface(surf_up_0, -1)
        cell.addSurface(surf_down_0)
        cell.addSurface(surf_outer_0, -1)
        self.cells.append(cell)

        exS = GeometricModel.MCNP_UnionSurface((surf_up_0, 1), (surf_down_0, -1), (surf_outer_0, 1))

        self._surface_exclude.append(exS)


energy_str = \
"si1   L 0.03987 0.04652 0.05323 0.0633 0.0844 0.0923 0.0928 0.0995\n\
        0.1291 0.1542 0.186 0.2094 0.23863 0.24098 0.24192 0.2703\n\
        0.27736 0.29522 0.30011 0.328 0.3384 0.35199 0.4094 0.463\n\
        0.51072 0.5623 0.58314 0.60937 0.6656 0.72227 0.7552 0.7684\n\
        0.7721 0.78546 0.7948 0.8062 0.8356 0.8402 0.86047 0.9111\n\
        0.934 0.9646 0.9689 1.1204 1.1553 1.2382 1.281 1.3777 1.408\n\
        1.4592 1.46 1.4958 1.5093 1.5879 1.62062 1.6304 1.6614 1.72\n\
        1.7646 1.8476 2.1187 2.2043 2.448 2.61447\n\
sp1     8.28474E-04 1.99922E-03 1.07275E-03 2.77453E-03 1.20505E-02\n\
        1.53599e-03 1.73103E-03 1.06195E-03 2.28207E-03 7.68221E-04\n\
        1.90170e-03 3.54737E-03 3.35909E-02 2.78669E-03 3.70587E-03\n\
        2.93732e-03 1.76239E-03 9.21591E-03 2.57580E-03 2.62099E-03\n\
        9.33916e-03 1.77004E-02 1.73980E-03 3.61516E-03 6.10058E-03\n\
        7.68221e-04 2.33479E-02 2.08699E-02 6.82660E-04 5.01603E-03\n\
        8.58600e-04 2.34055E-03 1.26531E-03 8.36006E-04 3.77332E-03\n\
        5.36376e-04 1.41594E-03 7.68221E-04 3.25364E-03 2.25947E-02\n\
        1.51160e-03 4.24781E-03 1.36322E-02 7.31421E-03 8.28944E-04\n\
        2.97445e-03 7.31421E-04 2.09674E-03 1.26780E-03 8.20942E-04\n\
        6.97031e-01 8.20942E-04 1.07275E-03 2.89213E-03 1.13727E-03\n\
        1.52138e-03 5.65633E-04 1.56037E-03 8.14316E-03 1.12151E-03\n\
        6.33899e-04 2.58436E-03 8.04564E-04 2.71137E-02"
energy_str = AuxiliaryFunction.line_merge(energy_str)
energy = SourceDefine.get_distribution_from_string(energy_str)

# 地层模型
class StratumInput(MCNPAutoInput.CellsPackage):
    def __init__(self, model:SingleStratumModel,parent:MCNPAutoInput.CellsPackage = None,transform:GeometricModel.MCNP_transformation = None):
        '''
        地层模型，基于model给出的地层情况来生成一个mcnp描述，通过get_cardsPool()来获取一个描述类
        :param model: 地层模型
        :param parent:
        :param transform:
        '''
        self.model = model
        self.independentSourceExt = SourceDefine.MCNP_F_Distribution("CELL")
        super().__init__(parent=parent, transform=transform)
        # Borehole(self.model.zone.innerDiameter, self.model.zone.Radius, parent=self, transform=self.transform)

        # 设置源类
        baseSource = SourceDefine.MCNP_SourceCards()
        baseSource.mode = 'p'
        baseSource["ERG"] = energy[0]  # energy为类外的全局变量
        baseSource["par"] = 2
        srange: dict = self.model.zone.get_SourceRange()
        for key, value in srange.items():
            baseSource[key] = value
        # 重新设置ext，让其依据cell来分布
        baseSource['ext'] = self.independentSourceExt
        self.baseSource = baseSource
        # 设置统计类
        self.tally = SourceDefine.MCNP_tallyCards()
        self.tally.type = 4
        self.tally.particle = baseSource.mode
        self.tally.energyBin.append(2.81)

    def create_cells(self):
        cell_basic = GeometricModel.MCNP_cell(material=Material.MaterialStorage().get_material_from_name("water_bearing_sand"), density=2.115)
        surf_up = GeometricModel.MCNP_PlaneSurface('py', [self.model.zone.topSurface], transformation=self.transform)
        surf_down = GeometricModel.MCNP_PlaneSurface('py', [self.model.zone.downSurface], transformation=self.transform)
        surf_outer = GeometricModel.MCNP_surface('cy', [self.model.zone.Radius], transformation=self.transform)
        cell_basic.addSurface(surf_up, -1)
        cell_basic.addSurface(surf_down, 1)
        cell_basic.addSurface(surf_outer, -1)

        # 创建模型中的地层
        for stb in self.model.stratumBorders:
            cell = GeometricModel.MCNP_cell(Material.MaterialStorage().get_material_from_name("oil_bearing_sand"),
                                            2.115, doseStand=stb.doseStand)
            for boundary_surf, d in stb.boundary.get_MCNP_Surface():
                boundary_surf.transformation = self.transform
                cell.addSurface(boundary_surf, d)
            cell_basic.addExcludeCell(cell)  # 将对应cell从背景中扣除
            self.cells.append(cell)
            # 对每一个有源地层进行设置源的范围
            up, down = stb.get_extRange()
            ext_si = SourceDefine.MCNP_SourceInformation()
            ext_sp = SourceDefine.MCNP_SourceProbabilty()
            ext_si.data.append(down)
            ext_si.data.append(up)
            ext_sp.data.append(-21)
            ext_sp.data.append(0)
            ext = SourceDefine.MCNP_Distribution_discrete(ext_si, ext_sp)
            self.independentSourceExt.add_distribution(ext)
        # 加入背景地层
        self.cells.append(cell_basic)
        # 设置当前体元边界
        exS = GeometricModel.MCNP_UnionSurface((surf_up, 1), (surf_down, -1), (surf_outer, 1))
        self._surface_exclude.append(exS)

    def get_cardsPool(self) -> MCNP_CardsPool:
        cardPool = super().get_cardsPool()
        cardPool.SourceCard = self.baseSource
        cardPool.TallyCard = self.tally
        return cardPool

class WWGCreatorInput(MCNPAutoInput.CellsPackage):
    def __init__(self, zone:DetectionZone, parent:MCNPAutoInput.CellsPackage = None, transform:GeometricModel.MCNP_transformation = None):
        self.zone = zone
        super().__init__(parent=parent, transform=transform)
        # 设置源
        baseSource = SourceDefine.MCNP_SourceCards()
        baseSource.mode = 'p'
        baseSource["ERG"] = energy[0]
        baseSource["par"] = 2
        srange: dict = self.zone.get_SourceRange()
        for key, value in srange.items():
            baseSource[key] = value
        # 普通的ext,整个地层设置为源
        si = SourceDefine.MCNP_SourceInformation()
        si.data.append(self.zone.downSurface)
        si.data.append(self.zone.topSurface)
        sp = SourceDefine.MCNP_SourceProbabilty()
        sp.data.append(-21)
        sp.data.append(0)
        self.SourceExt = SourceDefine.MCNP_Distribution_histogram(si, sp)
        baseSource['ext'] = self.SourceExt
        self.baseSource = baseSource
        self.tally = SourceDefine.MCNP_tallyCards()
        self.tally.type = 4
        self.tally.particle = baseSource.mode
        self.tally.energyBin.append(2.81)

    def create_cells(self):
        '''
        只需要创建背景地层
        :return:
        '''
        cell_basic = GeometricModel.MCNP_cell(material=Material.MaterialStorage().get_material_from_name("water_bearing_sand"), density=2.115, doseStand=1)
        surf_up = GeometricModel.MCNP_PlaneSurface('py', [self.zone.topSurface], transformation=self.transform)
        surf_down = GeometricModel.MCNP_PlaneSurface('py', [self.zone.downSurface], transformation=self.transform)
        surf_outer = GeometricModel.MCNP_surface('cy', [self.zone.Radius], transformation=self.transform)
        cell_basic.addSurface(surf_up, -1)
        cell_basic.addSurface(surf_down, 1)
        cell_basic.addSurface(surf_outer, -1)

        self.cells.append(cell_basic)
        exS = GeometricModel.MCNP_UnionSurface((surf_up, 1), (surf_down, -1), (surf_outer, 1))
        self._surface_exclude.append(exS)

    def get_cardsPool(self) -> MCNP_CardsPool:
        cardPool = super().get_cardsPool()
        cardPool.SourceCard = self.baseSource
        cardPool.TallyCard = self.tally
        return cardPool

    def get_wwgCardPool(self) -> list[MCNP_CardsPool]:
        cardPool = super().get_cardsPool()
        cardPool.SourceCard = self.baseSource
        cardPool.TallyCard = self.tally
        cellCards = deepcopy(cardPool.CellCards)
        r = []
        for d in cellCards:
            d.detector = False
        for i in range(len(cardPool.CellCards)):
            d = cardPool.CellCards[i]
            if d.detector:
                cp = deepcopy(cardPool)
                cd = deepcopy(cellCards)
                cd[i].detector = True
                cp.CellCards = cd
                r.append(cp)
        return r

# 用于测试用的地层模型输入类，只返回全局的地层模型
class Borehole_complete(MCNPAutoInput.CellsPackage):
    def __init__(self, boxRange:np.array,parent:MCNPAutoInput.CellsPackage = None,transform:GeometricModel.MCNP_transformation = None):
        self.centreX = boxRange[0]
        self.centreZ = boxRange[1]
        self.innerDiameter = boxRange[2]
        self.radius = boxRange[3]
        self.upY = boxRange[4]
        self.downY = boxRange[5]
        super().__init__(parent=parent, transform=transform)

    def create_cells(self):
        m = Material.MaterialStorage().get_material_from_name("H2O")
        cell = GeometricModel.MCNP_cell(m, 1.0, note='Borehole')

        surf_up_0 = GeometricModel.MCNP_PlaneSurface('py', [self.upY], 'Upper hole interface', transformation=self.transform)
        surf_down_0 = GeometricModel.MCNP_PlaneSurface('py', [self.downY], 'lower hold interface', transformation=self.transform)
        surf_outer_0 = GeometricModel.MCNP_surface('cy', [self.innerDiameter], 'Borehole boundary', transformation=self.transform)
        cell.addSurface(surf_up_0, -1)
        cell.addSurface(surf_down_0)
        cell.addSurface(surf_outer_0, -1)
        self.cells.append(cell)

        exS = GeometricModel.MCNP_UnionSurface((surf_up_0, 1), (surf_down_0, -1), (surf_outer_0, 1))

        self._surface_exclude.append(exS)

class StratumCompleteInput(MCNPAutoInput.CellsPackage):
    def __init__(self, stratumData:pd.DataFrame, api:np.array, boxRange:np.array,
                 parent:MCNPAutoInput.CellsPackage = None,
                 transform:GeometricModel.MCNP_transformation = None,
                 stratumAzi:np.array = None):
        '''

        :param stratumData:
        :param boxRange: 直井范围,[centreX, centreZ, innerDiameter,radius, upY, downY]
        :param parent:
        :param transform:
        '''
        self.mode = stratumData
        if stratumData.shape[0] > 3:
            raise AttributeError
        self.api = api
        self.Azi = stratumAzi

        self.centreX = boxRange[0]
        self.centreZ = boxRange[1]
        self.innerDiameter = boxRange[2]
        self.radius = boxRange[3]
        self.upY = boxRange[4]
        self.downY = boxRange[5]

        super().__init__(parent=parent, transform=transform)

        # 设置源
        baseSource = SourceDefine.MCNP_SourceCards()
        baseSource.mode = 'p'
        baseSource["ERG"] = energy[0]
        baseSource["par"] = 2

        rad_si = SourceDefine.MCNP_SourceInformation()
        rad_sp = SourceDefine.MCNP_SourceProbabilty()
        rad_si.data.append(self.innerDiameter)
        rad_si.data.append(self.radius)
        rad_sp.data.append(-21)
        rad_sp.data.append(1)
        rad = SourceDefine.MCNP_Distribution_discrete(rad_si, rad_sp)
        baseSource["rad"] = rad

        ext_si = SourceDefine.MCNP_SourceInformation()
        ext_sp = SourceDefine.MCNP_SourceProbabilty()
        ext_si.data.append(self.downY)
        ext_si.data.append(self.upY)
        ext_sp.data.append(-21)
        ext_sp.data.append(0)
        ext = SourceDefine.MCNP_Distribution_discrete(ext_si, ext_sp)
        baseSource['ext'] = ext

        axs = '0 1 0'
        pos = '{} 0 {}'.format(self.centreX, self.centreZ)
        baseSource['axs'] = axs
        baseSource['pos'] = pos

        self.baseSource = baseSource
        # 设置统计
        self.tally = SourceDefine.MCNP_tallyCards()
        self.tally.type = 4
        self.tally.particle = baseSource.mode
        self.tally.energyBin.append(2.81)

    def _create_cells(self):
        '''
        创建地层，范围为直井总体探测范围
        :return:
        '''
        # 沿着y轴从小到大建立
        upBoundary = []
        j = 0
        for i in range(self.mode.shape[0] - 1):
            controlPoint_0 = coordinate.Point(self.mode.index.values[i], self.mode.iloc[i, j], 0)
            controlPoint_1 = coordinate.Point(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 0)

            if self.Azi is None:
                # 使用右手系，下界面方向向量应该y轴负方向
                dipDirectionPoint = coordinate.Point(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 1)
            else:
                azi = self.Azi[i, j]
                _thisCoord = (coordinate.CoordinateSystem().move_to(
                    coordinate.ThreeDVector(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 0)).
                              rotate(coordinate.ThreeDVector(0, 1, 0), azi))
                dipDirectionPoint = coordinate.Point(0, 0, 1, _thisCoord).get_point_in_origin()
            plane = -geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint)
            upBoundary.append(plane)
        # y
        for j in range(1, self.mode.shape[1] + 1):
            cell = GeometricModel.MCNP_cell(material=Material.MaterialStorage().get_material_from_name("water_bearing_sand"),
                                            density=2.115, doseStand=self.api[j-1])
            downBoundary = []
            # x
            if j != self.mode.shape[1]:
                for i in range(self.mode.shape[0] - 1):
                    controlPoint_0 = coordinate.Point(self.mode.index.values[i], self.mode.iloc[i, j], 0)
                    controlPoint_1 = coordinate.Point(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 0)
                    if self.Azi is None:
                        # 使用右手系，下界面方向向量应该y轴负方向
                        dipDirectionPoint = coordinate.Point(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 1)
                    else:
                        azi = self.Azi[i, j]
                        _thisCoord = (coordinate.CoordinateSystem().move_to(
                            coordinate.ThreeDVector(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 0)).
                                      rotate(coordinate.ThreeDVector(0, 1, 0), azi))
                        dipDirectionPoint = coordinate.Point(0, 0, 1, _thisCoord).get_point_in_origin()
                    plane = geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint)
                    downBoundary.append(plane)
            # 如果是双平面则需要检查双平面
            if len(upBoundary) == 1:
                b = stratumModel.OnePlaneBoundary(upBoundary[0])
                for boundary_surf, d in b.get_MCNP_Surface():
                    boundary_surf.transformation = self.transform
                    cell.addSurface(boundary_surf, d)
            elif len(upBoundary) == 2:
                upB_1 = stratumModel.OnePlaneBoundary(upBoundary[0])
                upB_2 = stratumModel.OnePlaneBoundary(upBoundary[1])
                up_crossVec = np.cross(upBoundary[0].normal, upBoundary[1].normal)
                if np.dot(up_crossVec, np.array([0, 0, 1])) > 0:
                    upB = stratumModel.UnionAndPlanesBoundary(upB_1, upB_2)
                else:
                    upB = stratumModel.UnionOrPlanesBoundary(upB_1, upB_2)
                for boundary_surf, d in upB.get_MCNP_Surface():
                    boundary_surf.transformation = self.transform
                    cell.addSurface(boundary_surf, d)

            if len(downBoundary) == 0:
                suf = GeometricModel.MCNP_PlaneSurface('py', [self.upY], transformation=self.transform)
                cell.addSurface(suf, -1)
            elif len(downBoundary) == 1:
                b = stratumModel.OnePlaneBoundary(downBoundary[0])
                for boundary_surf, d in b.get_MCNP_Surface():
                    boundary_surf.transformation = self.transform
                    cell.addSurface(boundary_surf, d)
            elif len(downBoundary) == 2:
                downB_1 = stratumModel.OnePlaneBoundary(downBoundary[0])
                downB_2 = stratumModel.OnePlaneBoundary(downBoundary[1])
                down_crossVec = np.cross(lowerBoundary[0].normal, lowerBoundary[1].normal)
                if np.dot(down_crossVec, np.array([0, 0, 1])) > 0:
                    lowB = UnionOrPlanesBoundary(lowB_1, lowB_2)
                else:
                    lowB = UnionAndPlanesBoundary(lowB_1, lowB_2)

                for boundary_surf, d in lowB.get_MCNP_Surface():
                    boundary_surf.transformation = self.transform
                    cell.addSurface(boundary_surf, d)

            upBoundary = list(map(lambda x: -x, downBoundary))
            boundary_cy = GeometricModel.MCNP_surface('cy', [self.radius],
                                                      transformation=GeometricModel.MCNP_transformation(AuxiliaryFunction.transformationMatrix(),
                                                                                                        self.centreX, 0, self.centreZ))

            cell.addSurface(boundary_cy, -1)
            self.cells.append(cell)

            surf_up_0 = GeometricModel.MCNP_PlaneSurface('py', [self.upY], 'Upper hole interface', transformation=self.transform)
            surf_down_0 = GeometricModel.MCNP_PlaneSurface('py', [self.downY], 'lower hold interface', transformation=self.transform)
            surf_outer_0 = GeometricModel.MCNP_surface('cy', [self.radius], 'Borehole boundary', transformation=self.transform)

            exS = GeometricModel.MCNP_UnionSurface((surf_up_0, 1), (surf_down_0, -1), (surf_outer_0, 1))
            self._surface_exclude.append(exS)

    def create_cells(self):
        '''
        创建地层，范围为直井总体探测范围
        :return:
        '''
        # 沿着y轴从小到大建立
        upBoundary = []
        j = 0
        for i in range(self.mode.shape[0] - 1):
            controlPoint_0 = coordinate.Point(self.mode.index.values[i], self.mode.iloc[i, j], 0)
            controlPoint_1 = coordinate.Point(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 0)

            if self.Azi is None:
                # 使用右手系，下界面方向向量应该y轴负方向
                dipDirectionPoint = coordinate.Point(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 1)
            else:
                azi = self.Azi[i, j]
                _thisCoord = (coordinate.CoordinateSystem().move_to(
                    coordinate.ThreeDVector(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 0)).
                              rotate(coordinate.ThreeDVector(0, 1, 0), azi))
                dipDirectionPoint = coordinate.Point(0, 0, 1, _thisCoord).get_point_in_origin()
            plane = -geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint)
            upBoundary.append(plane)

        stratumList = []
        # y
        for j in range(1, self.mode.shape[1] + 1):
            downBoundary = []
            # x
            if j != self.mode.shape[1]:
                for i in range(self.mode.shape[0] - 1):
                    controlPoint_0 = coordinate.Point(self.mode.index.values[i], self.mode.iloc[i, j], 0)
                    controlPoint_1 = coordinate.Point(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 0)
                    if self.Azi is None:
                        # 使用右手系，下界面方向向量应该y轴负方向
                        dipDirectionPoint = coordinate.Point(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 1)
                    else:
                        azi = self.Azi[i, j]
                        _thisCoord = (coordinate.CoordinateSystem().move_to(
                            coordinate.ThreeDVector(self.mode.index.values[i + 1], self.mode.iloc[i + 1, j], 0)).
                                      rotate(coordinate.ThreeDVector(0, 1, 0), azi))
                        dipDirectionPoint = coordinate.Point(0, 0, 1, _thisCoord).get_point_in_origin()
                    plane = geometry.Plane(controlPoint_0, controlPoint_1, dipDirectionPoint)
                    downBoundary.append(plane)
            # 如果是双平面则需要检查双平面
            if len(upBoundary) == 1:
                upB = stratumModel.OnePlaneBoundary(upBoundary[0])
            elif len(upBoundary) == 2:
                upB_1 = stratumModel.OnePlaneBoundary(upBoundary[0])
                upB_2 = stratumModel.OnePlaneBoundary(upBoundary[1])
                up_crossVec = np.cross(upBoundary[0].normal, upBoundary[1].normal)
                if np.dot(up_crossVec, np.array([0, 0, 1])) > 0:
                    upB = stratumModel.UnionAndPlanesBoundary(upB_1, upB_2)
                else:
                    upB = stratumModel.UnionOrPlanesBoundary(upB_1, upB_2)

            if len(downBoundary) == 1:
                downB = stratumModel.OnePlaneBoundary(downBoundary[0])
            elif len(downBoundary) == 2:
                downB_1 = stratumModel.OnePlaneBoundary(downBoundary[0])
                downB_2 = stratumModel.OnePlaneBoundary(downBoundary[1])
                down_crossVec = np.cross(lowerBoundary[0].normal, lowerBoundary[1].normal)
                if np.dot(down_crossVec, np.array([0, 0, 1])) > 0:
                    lowB = UnionOrPlanesBoundary(lowB_1, lowB_2)
                else:
                    lowB = UnionAndPlanesBoundary(lowB_1, lowB_2)
            if len(downBoundary) > 0:
                B = UnionAndPlanesBoundary(upB, lowB)
            else:
                B = upB
            stratum = Stratum(B, self.api[j - 1], dZone)

            upBoundary = list(map(lambda x: -x, downBoundary))
            boundary_cy = GeometricModel.MCNP_surface('cy', [self.radius],
                                                      transformation=GeometricModel.MCNP_transformation(AuxiliaryFunction.transformationMatrix(),
                                                                                                        self.centreX, 0, self.centreZ))

            cell.addSurface(boundary_cy, -1)
            self.cells.append(cell)

            surf_up_0 = GeometricModel.MCNP_PlaneSurface('py', [self.upY], 'Upper hole interface', transformation=self.transform)
            surf_down_0 = GeometricModel.MCNP_PlaneSurface('py', [self.downY], 'lower hold interface', transformation=self.transform)
            surf_outer_0 = GeometricModel.MCNP_surface('cy', [self.radius], 'Borehole boundary', transformation=self.transform)

            exS = GeometricModel.MCNP_UnionSurface((surf_up_0, 1), (surf_down_0, -1), (surf_outer_0, 1))
            self._surface_exclude.append(exS)




    def get_cardsPool(self) -> MCNP_CardsPool:
        cardPool = super().get_cardsPool()
        cardPool.SourceCard = self.baseSource
        cardPool.TallyCard = self.tally
        return cardPool





