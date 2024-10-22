from MCNPInput import Material, SourceDefine
import numpy as np
import re
from abc import abstractmethod, ABCMeta

class MCNP_GeoObject(object):
    def __init__(self):
        self.index = -1

    def set_index(self, idx:int):
        self.index = idx

    def str_index(self) -> str:
        return str(self.index)

class MCNP_transformation(MCNP_GeoObject):
    def __init__(self, matrix:np.array, x:float = 0, y:float = 0, z:float = 0):
        '''
        实际MCNP需要输入的是一个旋转矩阵，这里可能将引入package：coordinate会更好，但是目前够用
        :param xTran: 绕x轴顺时针旋转的角度
        :param yTran: 绕y轴顺时针旋转的角度
        :param zTran: 绕z轴顺时针旋转的角度
        :param x: 位移向量的x
        :param y: 位移向量的y
        :param z: 位移向量的z
        '''
        super().__init__()
        if matrix.shape != (3, 3):
            raise AttributeError("matrix.shape should be (3, 3)")
        self.matrix = matrix
        self.point = np.array([x, y, z])

    def implyToMatrix(self, vector:np.array):
        '''
        让对应的向量被旋转，但是位移部分没有
        :param vector:
        :return:
        '''
        return vector.dot(self.matrix)

    def __str__(self):
        TMatrix = self.matrix
        trs = ""
        for i in range(3):
            for j in range(3):
                if TMatrix[i, j] == 1:
                    trs += "{} ".format(1)
                elif TMatrix[i, j] == 0:
                    trs += "{} ".format(0)
                else:
                    trs += "{:.4f} ".format(TMatrix[i, j])
        point = "{} {} {}".format(*self.point)
        ss = "Tr{} {} {}\n".format(self.index, point, trs)
        # ss = AuxiliaryFunction.line_split(ss)
        return ss

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if np.allclose(self.matrix, other.matrix) and np.allclose(self.point,other.point):
                return True
        return False

    def __hash__(self):
        return hash(tuple(self.params))

    def __add__(self, other):
        if isinstance(other, type(self)):
            _matrix = np.dot(self.matrix, other.matrix)
            _point = self.point + other.point
            return MCNP_transformation(_matrix, *_point)
        elif other is None:
            return self

    def __sub__(self, other):
        if isinstance(other,type(self)):
            _otherInvMatrix = np.linalg.inv(other.matrix)
            _matrix = np.dot(self.matrix, _otherInvMatrix)
            _point = self.point - other.point
            return MCNP_transformation(_matrix, *_point)

class MCNP_surface(MCNP_GeoObject):
    def __init__(self, shape:str, param:list, note:str="", transformation:MCNP_transformation = None):
        '''
        面元卡，面元一般由三个部分组成，1. 面元类型（即平面，圆柱面等）2.面元参数，不同的面元类型接受不同的参数，3.旋转，如果这个面需要旋转，则会添加一个旋转卡
        28 1 pz -25， 28是面元序号，1是旋转卡序号， pz是面元类型，-25是面元参数
        :param shape: MCNP中用于描述面的类型的字符串
        :param param: 面元的参数
        :param note: 这个面的注释
        :param transformation: Tr旋转卡
        '''
        super().__init__()
        self.note = note
        self.params = param
        self.shape = shape
        self.transformation = transformation

    def __str__(self):
        def tmpFunc(i):
            _ans = i
            if isinstance(i, float):
                _ans = round(i, 4)
                if _ans > 10000:
                    _ans = "{:.4E}".format(_ans)

            return str(_ans)
        p = " ".join(map(tmpFunc, self.params))
        if self.transformation is None:
            ss = "{} {} {} ${}\n".format(self.index, self.shape, p, self.note) if self.note != '' else "{} {} {}\n".format(self.index, self.shape, p)
        else:
            ss = "{} {} {} {} ${}\n".format(self.index, self.transformation.index, self.shape, p, self.note) if self.note != '' \
                else "{} {} {} {}\n".format(self.index, self.transformation.index, self.shape, p)
        # ss = AuxiliaryFunction.line_split(ss)
        return ss

    def __eq__(self, other):
        if isinstance(other, type(self)):
            p = " ".join(map(str,self.params))
            p2 = " ".join(map(str,other.params))
            if self.shape == other.shape and p == p2 and self.transformation == other.transformation:
                return True
        return False

    def __hash__(self):
        p = " ".join(self.params)
        return hash((p, self.shape, self.transformation))

    def addParam(self,param:str):
        self.params.append(param)

    def str_index(self, direct:int = 1) -> str:
        return"{:d}".format(direct * self.index)

    def get_negative(self) -> list[('MCNP_surface', int)]:
        return [(self, -1)]

class MCNP_PlaneSurface(MCNP_surface):
    def __init__(self, shape:str, param:list, note:str="", transformation:MCNP_transformation = None):
        _shape = shape.strip().lower()
        if _shape in ['px', 'pz', 'py', 'p']:
            _param = []
            for s in param:
                if isinstance(s, str):
                    _param.extend(s.split())
                else:
                    _param.append(s)
            if _shape in ['px', 'pz', 'py']:
                if len(param) > 1:
                    raise AttributeError("param error")
            else:
                if len(param) > 4:
                    raise AttributeError("param error")
                else:
                    n = float(np.linalg.norm(np.array(_param[:3])))
                    _param = list(map(lambda x: x / n, _param))
            super().__init__(_shape, _param, note,transformation)

    @property
    def normal(self) -> np.array:
        if self.shape == 'px':
            return np.array([1, 0, 0])
        elif self.shape == 'py':
            return np.array([0, 1, 0])
        elif self.shape == 'pz':
            return np.array([0, 0, 1])
        else:
            def f(p):
                if isinstance(p, str):
                    return eval(p)
                else:
                    return p
            _param = list(map(f, self.params))
            return np.array(_param[:3],dtype=float)

    def __eq__(self, other):
        if isinstance(other, type(self)) and self.shape == other.shape \
                    and np.allclose(np.cross(self.normal, other.normal), [0, 0, 0])\
                    and self.transformation == other.transformation\
                    and self.params[-1] == other.params[-1]:
            return True
        return False

class MCNP_UnionSurface(MCNP_GeoObject):
    def __init__(self, *args):
        super().__init__()
        self.surfList = []
        self.directList = []
        for surf, direct in args:
            if not (isinstance(surf, MCNP_surface | MCNP_UnionSurface) and isinstance(direct, int)):
                raise AttributeError()
            else:
                self.surfList.append(surf)
                self.directList.append(direct)

    def str_index(self, direct:int =1) -> str:
        dirSurfs = []
        for i in range(len(self.surfList)):
            dirSurfs.append(f"-{self.surfList[i].index}" if self.directList[i] < 0 else f"{self.surfList[i].index}")
        s = ":".join(dirSurfs)
        r = "({})" if direct > 0 else "-({})"
        return r.format(s)

    def get_surfs(self) -> list[MCNP_surface]:
        r_list = []
        for item in self.surfList:
            if isinstance(item, MCNP_surface):
                r_list.append(item)
            elif isinstance(item, MCNP_UnionSurface):
                r_list.extend(item.get_surfs())
        return r_list

    def get_negative(self) -> list[(MCNP_surface,int)]:
        r = []
        for i in range(len(self.surfList)):
            r.append((self.surfList[i], -self.directList[i]))
        return r

class MCNP_cell():
    def __init__(self, material:Material.MCNP_material, density:float, note:str = "", detector = False, doseStand = 0):
        '''
        栅元卡
        19  1  -1. 24 -25 -26 (-1:2:4)     $井眼流体
        栅元卡序号 材料卡序号 -(密度) 面元卡序号
        :param material: 栅元材料
        :param density: 密度
        :param note: 注释
        :param detector: 是否是探测器
        '''
        self.material = material
        self.density = density
        self.note = note
        self.surface:list[MCNP_surface | MCNP_UnionSurface] = []
        self.direct:list[int] = []
        self.excludeCell:list[MCNP_cell] = []
        self.index = -1
        self.detector = detector
        self.doseStand = doseStand
        self.sourceDefine = SourceDefine.MCNP_SourceCards()

    def __str__(self):
        sfi = []
        for i in range(len(self.surface)):
            _tmpSurf = self.surface[i]
            _tmpDirect = self.direct[i]
            sfi.append(_tmpSurf.str_index(_tmpDirect))
        for i in range(len(self.excludeCell)):
            _tmpCell = self.excludeCell[i]
            sfi.append(_tmpCell.str_exclude())
        str_surface = " ".join(sfi)
        if self.material.index == 0:
            ss = "{} 0 {} ${}\n".format(self.index,str_surface,self.note) if self.note != '' \
                else "{} 0 {}\n".format(self.index, str_surface)
        else:
            ss = "{} {} -{} {} ${}\n".format(self.index, self.material.index, self.density, str_surface, self.note) if self.note != '' \
                else "{} {} -{} {}\n".format(self.index, self.material.index, self.density, str_surface)
        # ss = AuxiliaryFunction.line_split(ss)
        return ss

    def addSurface(self, sur:MCNP_surface | MCNP_UnionSurface, direct:int = 1):
        '''
        添加一个面元卡
        :param sur: 面元卡的描述类
        :param direct: 方向
        :return:
        '''
        self.surface.append(sur)
        self.direct.append(direct)

    def addUnionSurface(self, *args,direct:int = 1):
        '''
        添加一个联合平面
        一个平面通过一个元组(sur:MCNP_surface, direct:int)来描述
        整个联合平面的方向需要另行给参数
        :param args:
        :param direct: 方向
        :return:
        '''
        uf = MCNP_UnionSurface(*args)
        self.addSurface(uf, direct)

    def addExcludeCell(self, cell:'MCNP_cell'):
        '''
        添加一个需要排除的体元（相当于布尔操作的减去）
        :param cell:
        :return:
        '''
        if isinstance(cell, type(self)):
            self.excludeCell.append(cell)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if self.material == other.material \
                    and len(self.surface) == len(other.surface)\
                    and self.density == other.density:
                for i in range(len(self.surface)):
                    s = self.surface[i]
                    d = self.direct[i]
                    if s not in other.surface:
                        return False
                    else:
                        idx = other.surface.index(s)
                        if d != other.direct[idx]:
                            return False
                return True
        return False

    def str_exclude(self) -> str:
        '''
        从一个体元中除去这个体元时，使用这个函数获取对应面情况
        :return:
        '''
        if len(self.excludeCell) == 0:
            U_surfaces= []
            for i in range(len(self.surface)):
                s = self.surface[i]
                if isinstance(s, MCNP_UnionSurface):
                    return "#{:d}".format(self.index)
                U_surfaces.append((self.surface[i], -self.direct[i]))
            return MCNP_UnionSurface(*U_surfaces).str_index()
        else:
            return "#{:d}".format(self.index)

    def get_surfs(self) -> list[MCNP_surface]:
        '''
        由于自身的list中存在联合面元的存在，通过这个函数得到面元列表
        :return:
        '''
        r = []
        for i in self.surface:
            if isinstance(i, MCNP_surface):
                r.append(i)
            elif isinstance(i, MCNP_UnionSurface):
                r.extend(i.get_surfs())
        return r

def createTransformationFromString(input:str) -> MCNP_transformation:
    params = input.split()
    point = []
    matrix = np.empty((3, 3))
    for i in range(1,4):
        point.append(eval(params[i]))
    for i in range(3):
        for j in range(3):
            _s = params[4+i*3+j]
            matrix[i,j] = eval(_s)
    return MCNP_transformation(matrix, *point)

def readTransformtionCard(input:str) -> list[MCNP_transformation]:
    trList = []
    lines = input.split('\n')
    inputList = re.compile(r'^tr\d+.*', re.IGNORECASE)
    for i in range(len(lines)):
        line = lines[i]
        if inputList.match(line):
            trList.append(createTransformationFromString(line))
    return trList

def createSurfaceFromString(input:str, trList:list[MCNP_transformation]) -> MCNP_surface:
    param = input.split()
    if re.match(r'\d+', param[1]):
        tr = eval(param[1]) - 1
        if param[2].lower() in ['px', 'pz', 'py', 'p']:
            s = MCNP_PlaneSurface(param[2], param[3:], transformation=trList[tr])
        else:
            s = MCNP_surface(param[2], param[3:], transformation=trList[tr])
    else:
        if param[2].lower() in ['px', 'pz', 'py', 'p']:
            s = MCNP_PlaneSurface(param[1], param[2:])
        else:
            s = MCNP_surface(param[1], param[2:])
    return s

def readSurfaceCard(input:str, trList:list[MCNP_transformation]) -> list[MCNP_surface]:
    surfaceList = []
    lines = input.split('\n')
    for i in range(len(lines)):
        line = lines[i]
        if len(line) > 0 and line[0].lower() != 'c':
            s = createSurfaceFromString(line, trList)
            surfaceList.append(s)
    return surfaceList

def createCellFromString(input:str, surfaceList:list[MCNP_surface], materialList:list[Material.MCNP_material], cellList:list[MCNP_cell]):
    param = input.split()
    matIdx = eval(param[1])
    if matIdx == 0:
        star = 2
        mat = Material.vacuo()
        density = 0
    else:
        star = 3
        mat = materialList[matIdx - 1]
        density = -eval(param[2])
    c = MCNP_cell(mat,density)
    for i in range(star,len(param)):
        p = param[i]
        if p[0] == '#':
            cellIdx =eval(p[1:]) - 1
            c.addExcludeCell(cellList[cellIdx])
        elif ":" in p:
            d = 1
            if p[0] == '-':
                p = p[1:]
                d = -1
            else:
                d = 1
            p = p.strip("()")
            arg = []
            p = p.split(':')
            for j in range(len(p)):
                surIdx = eval(p[j])
                if surIdx > 0:
                    surIdx = eval(p[j]) - 1
                    arg.append((surfaceList[surIdx], 1))
                else:
                    surIdx = -eval(p[j]) - 1
                    arg.append((surfaceList[surIdx], -1))
            c.addUnionSurface(*arg, direct=d)
        else:
            idx = eval(p)
            if idx > 0:
                c.addSurface(surfaceList[idx - 1])
            else:
                c.addSurface(surfaceList[-idx - 1], -1)
    return c

def readCellCardFromString(input:str,surfaceList:list[MCNP_surface], materialList:list[Material.MCNP_material]) -> list[MCNP_cell]:
    cellList = []
    lines = input.split('\n')
    for i in range(len(lines)):
        line = lines[i]
        if line[0].lower() != 'c':
            c = createCellFromString(line, surfaceList, materialList, cellList)
            cellList.append(c)
    return cellList
