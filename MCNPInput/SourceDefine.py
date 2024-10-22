import numpy as np
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt
import re
import logging
logger = logging.getLogger("myapp")

class MCNP_SourceDefinition(object):
    def __init__(self, **kwargs):
        '''
        源定义描述
        :param kwargs:
        # cell = None # 体元
        # par = None    # 粒子类型
        # erg = None # 粒子动能
        # tme = None # 粒子初始时间
        # dir = None # 粒子方向与vec向量的夹角
        # vec = None # 粒子方向的vec
        # nrm = None #
        # # 粒子初始位置采样范围
        # pos = None  # 参考基准点
        # rad = None # 初始位置到POS（基准点）或AXS（基准轴）的距离
        # ext = None # 初始位置沿着AXS轴相对POS的距离
        # X = None
        # Y = None
        # Z = None
        '''
        self.argument = kwargs

    def __str__(self):
        sdef_ = "sdef "
        d_idx = 1
        d_str = ''
        for key, value in self.argument.items():
            if isinstance(value, MCNP_Distribution) or isinstance(value, MCNP_F_Distribution):
                value.index = d_idx
                d_str += str(value)
                d_idx += value.count
                sdef_ += "{}={} ".format(key, value.name)
            else:
                sdef_ += "{}={} ".format(key, value)
        sdef_ += "\n"
        sdef_ += d_str
        return sdef_

    def __setitem__(self, key, value):
        self.argument[key] = value

    def __getitem__(self, item):
        return self.argument[item]

class MCNP_SourceInformation(object):
    def __init__(self):
        '''
        mcnp中si的描述
        对应distribution的x部分
        '''
        self.data = []
        self.option = ''
        self.index = -1

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        data = " ".join(list(map(str,self.data)))
        s = "SI{:d} {} {}\n".format(self.index, self.option, data)
        return s

class MCNP_SourceProbabilty(object):
    def __init__(self):
        '''
        mcnp中sp的描述
        对应distribution的y部分
        '''
        self.data = []
        self.option = ''
        self.index = -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        data = " ".join(list(map(str,self.data)))
        s = ("SP{:d} {} {}\n").format(self.index, self.option, data)
        return s

class MCNP_NormalSourceProbabilty(MCNP_SourceProbabilty):
    '''
    会自动归一化的分布值描述
    '''
    def __str__(self):
        data = np.array(list(map(float,self.data)))
        _data_sum = np.sum(data)
        data = data / _data_sum
        data_str = " ".join(list(map("{:.4f}".format,data)))
        s = ("SP{:d} {} {}\n").format(self.index, self.option, data_str)
        return s

class MCNP_Distribution(metaclass=ABCMeta):
    def __init__(self, si: MCNP_SourceInformation, sp: MCNP_SourceProbabilty):
        '''
        mcnp中的分布描述
        :param si: x
        :param sp: y
        '''
        self.si = si
        self.sp = sp
        self._index = -1

    def __str__(self):
        self.si.index = self.index
        self.sp.index = self.index
        s = '{}{}'.format(self.si, self.sp)
        return s

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value:int):
        self._index = value

    @property
    def name(self):
        return "d{}".format(self.index)

    def x(self) -> np.array:
        return np.array(self.si.data)

    def y(self) -> np.array:
        if self.sp[0] == -21:
            f = lambda x: np.power(abs(x), self.sp[1])
            return np.array(list(map(f, self.si.data)))
        else:
            return np.array(self.sp.data)
    @property
    def count(self) -> int:
        return 1


    @abstractmethod
    def plot(self):
        pass

class MCNP_F_Distribution(object):
    def __init__(self, targetName:str):
        '''
        mcnp中条件概率分布
        :param targetName:使用的条件分布名
        '''
        self.target = targetName
        self.DistributionSet:list[MCNP_Distribution] = []
        self._index = -1

    def __str__(self):
        subIdx = ''
        s1 = "DS{:d} S {}\n"
        subStr = ''
        for i in self.DistributionSet:
            subStr += str(i)
            subIdx += str(i.index) + ' '
        s1 = s1.format(self.index, subIdx.strip())
        return s1 + subStr


    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value:int):
        self._index = value
        for i in range(len(self.DistributionSet)):
            d = self.DistributionSet[i]
            d.index = i + self.index + 1

    @property
    def name(self):
        return "F{} d{:d}".format(self.target,self.index)

    @property
    def count(self) -> int:
        return 1+len(self.DistributionSet)

    def add_distribution(self, distribution: MCNP_Distribution):
        self.DistributionSet.append(distribution)

    def plot(self):
        for i in self.DistributionSet:
            i.plot()



class MCNP_Distribution_histogram(MCNP_Distribution):
    '''
    MCNP 直方图分布
    '''

    def plot(self):
        x = np.linspace(self.si.data[0], self.si.data[1], 100)

        if self.sp.data[0] == -21:
            y = np.power(np.abs(x), self.sp.data[1])

        plt.grid(True)
        plt.title(str(self.si).split()[0])

        plt.plot(x, y)
        plt.show()

class MCNP_Distribution_discrete(MCNP_Distribution):
    '''
    mcnp离散分布
    '''

    def plot(self):
        plt.bar(self.x(), self.y(), width=0.04)
        plt.grid(True)
        plt.title(str(self.si).split()[0])
        plt.show()

def get_distribution_from_string(input:str) -> list[MCNP_Distribution]:
    '''
    将mcnp字符串转换为分布描述对象
    :param input:
    :return:
    '''
    lines = input.split("\n")
    l_si = []
    l_sp = []
    l_distribution = []
    for i in range(len(lines)):
        line = lines[i]
        if re.match(r"^si\d+", line, re.IGNORECASE):
            s = line.split()
            si = MCNP_SourceInformation()
            si.index = int(s[0][2:])
            startIndex = 1
            if re.match(r'^[A-Z]', s[1],re.IGNORECASE):
                si.option = s[1]
                startIndex += 1
            for j in range(startIndex, len(s)):
                number_s = s[j]

                if re.search(r'[\.e]', number_s,re.IGNORECASE):
                    si.data.append(float(number_s))
                else:
                     si.data.append(int(number_s))
            l_si.append(si)
        elif re.match(r"^sp\d+", line, re.IGNORECASE):
            s = line.split()
            sp = MCNP_SourceProbabilty()
            sp.index = int(s[0][2:])
            startIndex = 1
            if re.match(r'^[A-Z]', s[1],re.IGNORECASE):
                sp.option = s[1]
                startIndex += 1
            for j in range(startIndex, len(s)):
                number_s = s[j]
                if re.search(r'[\.e]', number_s,re.IGNORECASE):
                    sp.data.append(float(number_s))
                else:
                    sp.data.append(int(number_s))
            l_sp.append(sp)
    for i in range(len(l_si)):
        si = l_si[i]
        sp = l_sp[i]
        if si.index == sp.index:
            if si.option == 'H' or si.option == '':
                l_distribution.append(MCNP_Distribution_histogram(si, sp))
            elif si.option == 'L':
                l_distribution.append(MCNP_Distribution_discrete(si, sp))
    return l_distribution

class MCNP_SourceCards(object):
    '''
    mcnp 源描述卡
    '''
    def __init__(self):
        self.mode = None
        self.imp = None
        self.sdef = MCNP_SourceDefinition()

    def __setitem__(self, key, value):
        self.sdef[key] = value

    def __getitem__(self, item):
        return self.sdef[item]

    def __str__(self):
        s = "mode {}\nimp:{} {}\n".format(self.mode, self.mode, self.imp)
        s += str(self.sdef)
        return s

    def set_universal_weight(self, cellNumbers:int):
        self.imp = "1 {:d}R 0".format(cellNumbers)

    def add_sourceCell(self, cell):
        self.cells.append(cell)

class MCNP_tallyCards(object):
    '''
    mcnp 统计卡
    '''
    def __init__(self):
        self.type = None
        self.particle = None
        self.detectors = []
        self.energyBin = []
        self.cut_nps = np.iinfo(np.int32).max

        self.tail = []

    def __str__(self):
        s = "f{}:{} ".format(self.type, self.particle)
        for d in self.detectors:
            s += "{:d} ".format(d.index)
        s += "\n"
        ebin = list(map(str, self.energyBin))
        e = "e{} {}\n".format(self.type, " ".join(ebin))
        cut = "nps {:d}\n".format(self.cut_nps)
        for t in self.tail:
            cut += str(t)
        return s + e + cut

    def add_detector(self, detector):
        self.detectors.append(detector)

def readSourceCardFromString(input:str) ->MCNP_SourceCards:
    '''
    将mcnp输入文件中的源描述卡部分转换为python描述对象
    :param input:
    :return:
    '''
    lines = input.split('\n')
    _mode = re.compile('^mode.*')
    _imp = re.compile('^imp.*')
    _sdef = re.compile('^sdef.*')
    source = MCNP_SourceCards()
    d = get_distribution_from_string(input)
    for line in lines:
        if _mode.match(line):
            source.mode = line.split()[1]
        if _imp.match(line):
            # source.imp = line.split(':')[1]
            continue
        if _sdef.match(line):
            c = re.compile(r"(\s+\w+)=")
            r = c.findall(line)
            arg = {}
            for i in range(len(r) - 1):
                _s = line[line.find(r[i]):line.find(r[i + 1])].strip()
                key, value = _s.split('=')
                value:str = value.strip()
                if value[0].lower() == 'd':
                    _idx:int = eval(value[1:]) - 1
                    value = d[_idx]
                arg[key] = value
            s_def = MCNP_SourceDefinition(**arg)
            source.sdef = s_def
    return source

class MCNP_wwgCard(object):
    def __init__(self, parent:MCNP_tallyCards):
        self.parent = parent
        self.invokeCell = 0
        self.weight_window = 0
        self.energyOrTime = 0
        self.meshArg = {}
        self.splitArg = []

    def __str__(self):
        i_t = self.parent.type
        i_c = self.invokeCell
        w_g = self.weight_window
        i_e = self.energyOrTime

        wwg_s = f'wwg {i_t} {i_c} {w_g} J J J J {i_e}\n'

        mesh_arg = []
        for key, value in self.meshArg.items():
            mesh_arg.append("{}={}".format(key, value))

        mesh = "mesh " + " ".join(mesh_arg) + "\n"

        if len(self.splitArg) == 6:
            head = ['imesh', 'iints', 'jmesh', 'jints', 'kmesh', 'kints']
            for i in range(6):
                mesh += "     {} {}\n".format(head[i],self.splitArg[i])

        return wwg_s + mesh







