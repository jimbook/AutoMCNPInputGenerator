from MCNPInput import MCNPAutoInput, GeometricModel, Material
from utility.readSCC import smallModel

class baseDetector(MCNPAutoInput.CellsPackage):
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

class StratumInput(MCNPAutoInput.CellsPackage):
    def __init__(self, model:smallModel,parent:MCNPAutoInput.CellsPackage = None,transform:GeometricModel.MCNP_transformation = None):
        self.model = model
        super().__init__(parent=parent, transform=transform)
        Borehole(self.model.zone.innerDiameter, self.model.zone.Radius, parent=self, transform=self.transform)

    def create_cells(self):
        cell_basic = GeometricModel.MCNP_cell(material=Material.MaterialStorage().get_material_from_name("water_bearing_sand"), density=2.115)
        surf_up = GeometricModel.MCNP_PlaneSurface('py', [self.model.zone.Radius], transformation=self.transform)
        surf_down = GeometricModel.MCNP_PlaneSurface('py', [-self.model.zone.Radius], transformation=self.transform)
        surf_outer = GeometricModel.MCNP_surface('cy', [self.model.zone.Radius], transformation=self.transform)
        cell_basic.addSurface(surf_up, -1)
        cell_basic.addSurface(surf_down, 1)
        cell_basic.addSurface(surf_outer, -1)

        for stb in self.model.stratumBorder:
            cell = GeometricModel.MCNP_cell(Material.MaterialStorage().get_material_from_name("oil_bearing_sand"),
                                            2.115)
            for boundary_surf, d in stb.boundary.get_MCNP_Surface():
                cell.addSurface(boundary_surf,d)
            cell_basic.addExcludeCell(cell)
            self.cells.append(cell)

        self.cells.append(cell_basic)


        exS = GeometricModel.MCNP_UnionSurface((surf_up, 1), (surf_down, -1), (surf_outer, 1))
        self._surface_exclude.append(exS)






