from . import AuxiliaryFunction, GeometricModel, Material, SourceDefine
from abc import abstractmethod, ABCMeta


class MCNP_CardsPool(object):
    def __init__(self):
        self.SurfaceCards:list[GeometricModel.MCNP_surface] = []
        self.CellCards:list[GeometricModel.MCNP_cell] = []
        self.MaterialCards:list[Material.MCNP_material] = []
        self.TransformationCards:list[GeometricModel.MCNP_transformation] = []
        self.SourceCard:SourceDefine.MCNP_souceCards = SourceDefine.MCNP_souceCards()
        self.TallyCard = SourceDefine.MCNP_tallyCards()

        self.sourceCells:list[GeometricModel.MCNP_cell] = []
        self.nps = pow(2, 32)

    def add_transformation(self, tr:GeometricModel.MCNP_transformation) -> int:
        '''
        添加一个旋转卡，一般不单独使用
        :param tr:
        :return: 面元卡的id
        '''
        if tr in self.TransformationCards:
            tr.index = self.TransformationCards.index(tr) + 1
        else:
            tr.index = len(self.TransformationCards) + 1
            self.TransformationCards.append(tr)
        return tr.index

    def add_material(self, mat: Material.MCNP_material) -> int:
        '''
        添加一个材料卡，一般不单独使用
        :param mat:
        :return:
        '''
        if mat in self.MaterialCards:
            mat.index = self.MaterialCards.index(mat) + 1
        else:
            mat.index = len(self.MaterialCards) + 1
            self.MaterialCards.append(mat)
        return mat.index

    def add_surface(self, surf:GeometricModel.MCNP_surface) -> int:
        '''
        添加一个面元，一般不单独使用
        :param surf:
        :return:
        '''
        if surf in self.SurfaceCards:
            surf.index = self.SurfaceCards.index(surf) + 1
        else:
            surf.index = len(self.SurfaceCards) + 1
            self.SurfaceCards.append(surf)
            if surf.transformation is not None:
                self.add_transformation(surf.transformation)
        return surf.index

    def add_cell(self, cell:GeometricModel.MCNP_cell):
        '''
        添加一个体元
        :param cell:
        :return:
        '''
        if cell in self.CellCards:
            cell.index = self.CellCards.index(cell) + 1
        else:
            cell.index = len(self.CellCards) + 1
            self.CellCards.append(cell)
            for surface in cell.get_surfs():
                self.add_surface(surface)
            if cell.material.index != 0:
                self.add_material(cell.material)
        if cell.detector:
            self.TallyCard.add_detector(cell)
        if cell.api != 0:
            self.sourceCells.append(cell)

    def _total_api(self) -> float:
        _a = 0.
        for cell in self.sourceCells:
            _a += cell.api
        return _a

    def __str__(self):
        cell_head = "vertical well of Azimuth gamma\nC ==============cell card================\n"
        surf_head = "C =============surface card==============\n"
        material_head = "C ===============material card==============\n"
        tran_head = "C =============transformation card================\n"
        source_head = "C ==============source card===================\n"
        tally_head = "C ==============tally card=================\n"

        cell_output = cell_head
        for cell in self.CellCards:
            cell_output += str(cell)
        cell_output = AuxiliaryFunction.line_split(cell_output)

        surf_output = surf_head
        for surf in self.SurfaceCards:
            surf_output += str(surf)
        surf_output = AuxiliaryFunction.line_split(surf_output)

        data_output = material_head
        for mat in self.MaterialCards:
            data_output += str(mat)
        data_output += tran_head
        for tr in self.TransformationCards:
            data_output += str(tr)
        data_output += source_head
        data_output += str(self.SourceCard)
        data_output += tally_head
        data_output += str(self.TallyCard)
        data_output = AuxiliaryFunction.line_split(data_output)


        output = "\n\n".join([cell_output, surf_output, data_output])
        return output

    def read_base_cards(self, path:str):
        with open(path, 'r', encoding='utf-8') as file:
            l = file.readline()
            s = AuxiliaryFunction.line_merge(file.read())
        cards = s.split('\n\n')
        trCards = GeometricModel.readTransformtionCard(cards[2])
        matCards = Material.readMaterialCard(cards[2])
        surfCards = GeometricModel.readSurfaceCard(cards[1], trCards)
        cellCards = GeometricModel.readCellCardFromString(cards[0], surfCards, matCards)
        for cell in cellCards:
            self.add_cell(cell)
        source = SourceDefine.readSourceCardFromString(cards[2])
        tally = SourceDefine.MCNP_tallyCards()
        tally.type = source.mode
        tally.energyBin.append(2.81)
        self.SourceCard = source
        self.TallyCard = tally

    def read_base_Geo(self, path:str):
        with open(path, 'r', encoding='utf-8') as file:
            l = file.readline()
            s = AuxiliaryFunction.line_merge(file.read())
        cards = s.split('\n\n')
        trCards = GeometricModel.readTransformtionCard(cards[2])
        matCards = Material.readMaterialCard(cards[2])
        surfCards = GeometricModel.readSurfaceCard(cards[1], trCards)
        cellCards = GeometricModel.readCellCardFromString(cards[0], surfCards, matCards)
        for cell in cellCards:
            self.add_cell(cell)

    def str_geo(self) -> str:
        cell_head = "vertical well of Azimuth gamma\nC ==============cell card================\n"
        surf_head = "C =============surface card==============\n"
        material_head = "C ===============material card==============\n"
        tran_head = "C =============transformation card================\n"

        cell_output = cell_head
        for cell in self.CellCards:
            cell_output += str(cell)
        cell_output = AuxiliaryFunction.line_split(cell_output)

        surf_output = surf_head
        for surf in self.SurfaceCards:
            surf_output += str(surf)
        surf_output = AuxiliaryFunction.line_split(surf_output)

        data_output = material_head
        for mat in self.MaterialCards:
            data_output += str(mat)
        data_output += tran_head
        for tr in self.TransformationCards:
            data_output += str(tr)
        data_output = AuxiliaryFunction.line_split(data_output)

        output = "\n\n".join([cell_output, surf_output, data_output])
        return output

class CellsPackage(metaclass=ABCMeta):
    def __init__(self, parent:'CellsPackage' = None,transform:GeometricModel.MCNP_transformation = None):
        self.cells:list[GeometricModel.MCNP_cell] = []
        self.transform = transform
        self._surface_exclude: list[(GeometricModel.MCNP_surface, int)] = []

        self.parent = parent
        self.children:list['CellsPackage'] = []
        if parent is not None:
            parent.add_child(self)

        self.create_cells()


    @abstractmethod
    def create_cells(self):
        pass

    def surface_exclude(self) -> list[(GeometricModel.MCNP_surface, int)]:
        return self._surface_exclude

    def add_child(self, child:'CellsPackage'):
        self.children.append(child)

    def push_cells(self, cardsPool:MCNP_CardsPool):
        for child in self.children:
            child.push_cells(cardsPool)
            for exS in child.surface_exclude():
                for cell in self.cells:
                    cell.addSurface(exS)
        for cell in self.cells:
            cardsPool.add_cell(cell)
        if self.parent is None:
            vacouCell = GeometricModel.MCNP_cell(Material.vacuo(), 0., 'vacou')
            for exS in self.surface_exclude():
                vacouCell.addSurface(exS)
            cardsPool.add_cell(vacouCell)

    def get_cardsPool(self) -> MCNP_CardsPool:
        cardsPool = MCNP_CardsPool()
        self.push_cells(cardsPool)
        return cardsPool


