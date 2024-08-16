import threading
import re

class MaterialStorage(object):
    _instance_lock = threading.Lock()

    def init(self):
        self._check = 0
        self._materialDic = {}
        self._materialList: list['MCNP_material'] = []

    def __new__(cls, *args, **kwargs):
        if not hasattr(MaterialStorage, "_instance"):
            with MaterialStorage._instance_lock:
                if not hasattr(MaterialStorage, "_instance"):
                    MaterialStorage._instance = object.__new__(cls)
                    MaterialStorage._instance.init()
        return MaterialStorage._instance

    def add_material(self, material:'MCNP_material'):
        self._check += 1
        if material.name not in self._materialDic.keys():
            idx = len(self._materialList)
            self._materialDic[material.name] = idx
            self._materialList.append(material)
        else:
            idx = self._materialDic[material.name]
            self._materialList[idx] = material

    def get_material_from_name(self, name:str):
        if name in self._materialDic.keys():
            idx = self._materialDic[name]
            return self._materialList[idx]
        else:
            return None

    def get_material_from_index(self, index:int):
        return self._materialList[index]

class MCNP_material():
    '''
    生成MCNP材料卡的对象
    '''
    def __init__(self, name: str = "", noName:bool = False):
        '''
        :param name: 材料名
        '''
        self.element:list[int] = [] # 内含的元素，
        self.content:list[int|float] = [] # 元素含量比例
        self.secLib:list[str|None] = [] # 指定核数据库
        self.name = name
        self._index = -1 # 在

        self._noName = noName
        if not noName:
            _m = MaterialStorage()
            _m.add_material(self)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value:int):
        self._index = value

    @property
    def noName(self) -> bool:
        return self._noName

    def add_element(self, element: int, content, secLib: str = None):
        self.element.append(element)
        self.content.append(content)
        self.secLib.append(secLib)
        if not self.noName:
            _m = MaterialStorage()
            _m.add_material(self)

    def set_name(self, name:str):
        if self.noName:
            self.name = name
            _m = MaterialStorage()
            _m.add_material(self)
            self._noName = False
        else:
            raise AttributeError("Can't set name if this class has had name.")


    def __str__(self) -> str:
        includeStr = ""
        if self.index == 0:
            return includeStr
        for i in range(len(self.element)):
            if self.secLib[i] is None:
                ss = "{} {}".format(self.element[i], self.content[i])
            else:
                ss = "{}.{} {}".format(self.element[i], self.secLib[i], self.content[i])
            if i == 0:
                includeStr += "{} ${}\n".format(ss, self.name)
            else:
                includeStr += "       {}\n".format(ss)
        return "m{} {}".format(self.index, includeStr)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            # 如果均为真空对象
            if self.index == 0 and other.index == 0:
                return True
            elif self.element == other.element and \
                    self.content == other.content and \
                    self.secLib == other.secLib:
                return True
        return False

class vacuo(MCNP_material):
    '''
    真空材料类，比较特殊，所以作为子类继承
    '''
    def __init__(self):
        super().__init__('vacuo')
        self._index = 0

    @property
    def index(self):
        return super().index

    @index.setter
    def index(self, values):
        raise AttributeError("index of vacuo can't be changed.")

    def addElement(self, element: int, content, secLib: str = None):
        raise AttributeError("vacuo can't have any element.")

def create_material_from_string(input:str) -> MCNP_material:
    l_str = input.split()
    m = MCNP_material(noName=True)

    _match_index = re.compile(r'^m\d+')
    _match_name = re.compile(r'^\$.*')
    # _match_element = re.compile(r'^\d+\.?')

    idx = 0
    while idx < len(l_str):
        s = l_str[idx]
        idx += 1
        if _match_index.match(s):
            continue
        if _match_name.match(s):
            if m.noName:
                m.set_name(s[1:])
        else:
            i = s.find(".")
            if i == -1:
                e = int(s)
                secLib = None
            else:
                e = int(s[:i])

                secLib = s[i:] if i < len(s) - 1 else None
            s = l_str[idx]
            idx += 1
            ic_0 = s.find('.')
            ic_1 = s.find('e')
            if ic_0 != -1 or ic_1 != -1:
                c = float(s)
            else:
                c = int(s)
            m.add_element(e, c, secLib)

    return m

def init_a_inner_material_database():
    vacuo()
    create_material_from_string("m1  1001  2        $H2O\
     8016  1")
    create_material_from_string('m2  11000  1      $NaI\
       53000 1')
    create_material_from_string('m3  6000  5        $rubber\
       1001 8')
    create_material_from_string('m4  6000  201     $epoxy_resin\
       1001  224\
       8016  34')
    create_material_from_string('m5  4009  -0.02     $Be-Cu\
      13027  -0.001\
      26056  -0.00105\
      82207  -5e-005\
      14000  -0.001\
      12000  -0.001\
      28058  -0.004\
      29063   -0.9719')
    create_material_from_string('m6  26054  -0.05845   $W-Ni-Fe\
       26056  -0.91754\
       26057  -0.02119\
       26058  -0.00282')
    create_material_from_string('m7  1001  -0.0154679   $water_bearing_sand\
       8016  -0.5828306\
       14000   -0.4017015')
    create_material_from_string('\
m8  1001.       -0.01064854   $oil_bearing_sand\
       6000.       -0.03048483  8016.        -0.4455387  11023.     -0.000347447\
      12000.      -0.06013266  13027.       -0.1013145  14000.        -0.265512\
      19000.      -0.01132272  20000.     -0.000868618 26056.      -0.07307289\
      25055.     -0.000757053')

def readMaterialCard(input:str) -> list[MCNP_material]:
    materialList = []
    lines = input.split('\n')
    inputList = re.compile(r'^m\d+.*')
    for i in range(len(lines)):
        line = lines[i]
        if inputList.match(line):
            materialList.append(create_material_from_string(line))
    return materialList

init_a_inner_material_database()
