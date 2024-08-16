import numpy as np
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt
import re

class MCNP_SourceDefinition(object):
    def __init__(self, **kwargs):
        # self._cell = None # 体元
        # self._par = None    # 粒子类型
        # self._energy = None # 粒子动能
        # self._time = None # 粒子初始时间
        # self._direction = None # 粒子方向与vec向量的夹角
        # self._vector = None # 粒子方向的vec
        # self._nrm = None #
        # # 粒子初始位置采样范围
        # self._pos = None  # 参考基准点
        # self._rad = None # 初始位置到POS（基准点）或AXS（基准轴）的距离
        # self._ext = None # 初始位置沿着AXS轴相对POS的距离
        # self._X = None
        # self._Y = None
        # self._Z = None
        # self._CCC = None
        # self._ARA = None
        # self._weight = None
        # self._transformation = None
        # self._efficiency = None
        # # 宇宙射线
        # self._dat = None
        # self._loc = None
        # self._bem = None
        # self._bar = None
        self.argument = kwargs

    def __str__(self):
        sdef_ = "sdef "
        d_ = []
        for key, value in self.argument.items():
            if isinstance(value, MCNP_Distribution):
                d_.append(value)
                value.index = len(d_)
                sdef_ += "{}={} ".format(key, value.name)
            else:
                sdef_ += "{}={} ".format(key, value)
        sdef_ += "\n"
        for d in d_:
            sdef_ += str(d)
        return sdef_


    def __setitem__(self, key, value):
        self.argument[key] = value

    def __getitem__(self, item):
        return self.argument[item]

class MCNP_SourceInformation(object):
    def __init__(self):
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

class MCNP_Distribution(metaclass=ABCMeta):
    def __init__(self, si: MCNP_SourceInformation, sp: MCNP_SourceInformation):
        self.si = si
        self.sp = sp
        self.index = -1

    def __str__(self):
        s = '{}{}'.format(self.si, self.sp)
        return s

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


    @abstractmethod
    def plot(self):
        pass

class MCNP_Distribution_histogram(MCNP_Distribution):
    # def __init__(self, si: MCNP_SourceInformation, sp: MCNP_SourceInformation):
    #     if len(si) - len(sp) != 1:
    #         raise AttributeError("len(si) - len(sp) != 1")
    #     else:
    #         super().__init__(si, sp)

    def plot(self):
        x = np.linspace(self.si.data[0], self.si.data[1], 100)

        if self.sp.data[0] == -21:
            y = np.power(np.abs(x), self.sp.data[1])

        plt.grid(True)
        plt.title(str(self.si).split()[0])

        plt.plot(x, y)
        plt.show()

class MCNP_Distribution_discrete(MCNP_Distribution):
    # def __init__(self, si: MCNP_SourceInformation, sp: MCNP_SourceInformation):
    #     if len(si) - len(sp) != 0:
    #         raise AttributeError("len(si) - len(sp) != 0")
    #     else:
    #         super().__init__(si, sp)

    def plot(self):
        plt.bar(self.x(), self.y(), width=0.04)
        plt.grid(True)
        plt.title(str(self.si).split()[0])
        plt.show()

def get_distribution_from_string(input:str) -> list[MCNP_Distribution]:
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

class MCNP_souceCards(object):
    def __init__(self):
        self.mode = None
        self.imp = None
        self.sdef = None
        self.cells = []

    def __str__(self):
        _l = len(self.cells)
        if _l == 2:
            imp = '1 0'
        elif _l > 2:
            imp = '1 {:}r 0'.format(_l - 2)
        else:
            imp = ''
        s = "mode {}\nimp:{} {}\n".format(self.mode, self.mode, imp)
        s += str(self.sdef)
        return s

    def set_universal_weight(self, cellNumbers:int):
        self.imp = "1 {:d}R 0".format(cellNumbers)

    def add_sourceCell(self, cell):
        self.cells.append(cell)

class MCNP_tallyCards(object):
    def __init__(self):
        self.type = None
        self.particle = None
        self.detectors = []
        self.energyBin = []
        self.cut_nps = np.power(2, 32)

    def __str__(self):
        s = "f{}:{} ".format(self.type, self.particle)
        for d in self.detectors:
            s += "{:d} ".format(d.index)
        s += "\n"
        ebin = list(map(str, self.energyBin))
        e = "e{} {}\n".format(self.type, " ".join(ebin))
        cut = "nps {:d}\n".format(self.cut_nps)
        return s + e + cut


    def add_detector(self, detector):
        self.detectors.append(detector)

def readSourceCardFromString(input:str):
    lines = input.split('\n')
    _mode = re.compile('^mode.*')
    _imp = re.compile('^imp.*')
    _sdef = re.compile('^sdef.*')
    source = MCNP_souceCards()
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



