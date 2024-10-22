import copy
import re
import io
import time

import pandas as pd
import numpy as np
from copy import deepcopy
from time import gmtime, strftime

# 读取output文件
class MCNP_OutputReader(object):
    def __init__(self, filename:str, depth:float = None):
        self.data = [] # 各个探测器的数据
        self.error = [] # 各个探测器数据的不确定度
        self.column = [] # 各个探测器的cell序号
        self.depth = depth # 当前探测点的探测深度
        with open(filename, 'r') as file:
            s = file.readline()
            _reMatch = re.compile(r'^1tally\s+4')
            while s:
                _m = _reMatch.match(s)
                if _m:
                    self.readTally4(file)
                    break
                else:
                    s = file.readline()

    def readTally4(self, file):
        '''
        读取tally4统计数据
        :param file:
        :return:
        '''
        _reCell = re.compile(r'^\s+cell\s+(\d+)')
        _reEnd = re.compile(r'^\s+=+')
        _reData = re.compile(r'^\s+.*?\s+(.*)')
        s = file.readline()
        while s:
            _m = _reCell.match(s)
            if _m:
                file.readline()
                d = file.readline()
                _d = _reData.match(d)
                if _d:
                    _d_num = _d.group(1).strip()
                    _d_num = _d_num.split()
                    self.column.append(int(_m.group(1)))
                    self.data.append(float(_d_num[0]))
                    self.error.append(float(_d_num[1]))
            _e = _reEnd.match(s)
            if _e:
                break
            s = file.readline()

# 读取wwout文件
class MCNP_WWOUTReader(object):
    def __init__(self, fileName:str):
        # 把文件的头存储下来，用于方便转存时直接写入头
        head = []
        with open(fileName, 'r') as file:
            s = ''
            while s.strip() != "100.00":
                head.append(s)
                s = file.readline()
            head.append(s)
        self.head = "".join(head[1:])

        # 读取内容
        with open(fileName, 'r') as file:
            # 第一行和第二行是表明权重网格属性
            s = file.readline()
            p = s.split()
            self.arg_if, self.arg_iv, self.arg_ni, self.arg_nr = map(int, p[:4])
            # arg_probid = p[4] + ' ' + p[5]
            # print(self.arg_if, self.arg_iv, self.arg_ni, self.arg_nr, arg_probid)
            if self.arg_if == 2:
                s = file.readline()
                p = s.split()
                self.arg_nt = np.array(map(float, p))
            s = file.readline()
            p = s.split()
            self.arg_ne = np.array(map(float, p))

            # 读取网格描述，并基于网格描述，生成每个格子的中心坐标（对于圆柱，这里使用r、z、theta圆柱坐标）
            p = []
            if self.arg_nr == 16:
                while len(p) != 16:
                    s = file.readline().strip()
                    tmp_p = s.split()
                    p.extend(map(float, tmp_p))
                self.arg_nf = np.array(p[:3])
                self.arg_v0 = np.array((p[3:6]))
                self.arg_nc = np.array(p[6:9])
                self.arg_v1 = np.array(p[9:12])
                self.arg_v2 = np.array(p[12:15])
                self.arg_nwg = p[15]

            grid_x = MCNP_WWOUTReader._read_grid(file)
            grid_y = MCNP_WWOUTReader._read_grid(file)
            grid_z = MCNP_WWOUTReader._read_grid(file)

            _mesh_grid_y, _mesh_grid_z, _mesh_grid_x = np.meshgrid(grid_y, grid_z, grid_x)
            mesh_grid_x = _mesh_grid_x.flatten()
            mesh_grid_y = _mesh_grid_y.flatten()
            mesh_grid_z = _mesh_grid_z.flatten()

            self.mesh_grid = np.vstack((mesh_grid_x, mesh_grid_y, mesh_grid_z)).T

            # 检查数据标志
            s = file.readline().strip()
            p = float(s)
            if not np.isclose(p, 100.):
                raise Exception("file structure error")

            # 读取数据
            p = []
            while s:
                s = file.readline()
                tmp_p = s.split()
                p.extend(map(float, tmp_p))

            self.data = np.array(p)
            self._points = None

    @property
    def points(self) -> np.array:
        '''
        返回直角坐标系下的坐标
        :return:
        '''
        if self._points is None:
            def to_point(v: np.array) -> np.array:
                r = v[0]
                z = v[1] + self.arg_v0[1]
                theta = v[2] * 2 * np.pi
                x = np.cos(theta) * r
                y = -np.sin(theta) * r
                return np.array([x, z, y])
            self._points = np.apply_along_axis(to_point, 1, self.mesh_grid)
        return self._points

    @staticmethod
    def _read_grid(file):
        p = []
        mesh_grid_x = []
        while len(p) % 3 == 0:
            s = file.readline().strip()
            tmp_p = s.split()
            p.extend(map(float, tmp_p))
        tmp_start = 0.
        for i in range(int(len(p) / 3)):
            tmp_x_p = p[i * 3:(i + 1) * 3]
            # tmp_start = tmp_x_p[0]
            tmp_nums = int(tmp_x_p[1]) + 2
            tmp_stop = tmp_x_p[2]
            grid = np.linspace(tmp_start, tmp_stop, tmp_nums)
            mesh_grid_x.extend(grid.tolist()[1:-1])
            tmp_start = tmp_stop

        mesh_grid_x = np.array(mesh_grid_x)
        return mesh_grid_x

    def save(self, path:str):
        h = self.head
        d = np.copy(self.data)
        _dl = []
        _n = []
        for i in d:
            if i == 0:
                _n.append("0.0000 ".center(13))
            elif i >= 1.0e6 or i < 1.0e-1:
                _n.append("{:.5E} ".format(i).center(13))
            else:
                _n.append("{:.5} ".format(i).center(13))
            if len(_n) == 6:
                _dl.append("".join(_n))
                _n.clear()
        if len(_n) != 0:
            _dl.append("".join(_n))
            _n.clear()
        h += "\n".join(_dl)
        with open(path, 'w') as file:
            file.write(h)

def merge_wwout(*args:MCNP_WWOUTReader) :
    _f = args[0]
    length = len(_f.data)
    r = []
    for out in args:
        if len(out.data) == length:
            _d = np.copy(out.data)
            _d[_d == 0.] = np.nan
            r.append(_d)

    r = np.vstack(r)
    m = r.mean(axis=1)
    wwout = np.nan_to_num(m)
    f_reader = deepcopy(_f)
    f_reader.data = wwout
    return f_reader

def merge_output(*arg:MCNP_OutputReader):
    _tmp = arg[0]

    sd = np.array(_tmp.data)
    se = np.power(np.array(_tmp.error) * sd, 2)

    for i in arg[1:]:
        od = np.array(i.data)
        oe = np.power(np.array(i.error) * od, 2)

        sd += od
        se += oe

    r = copy.deepcopy(_tmp)
    r.data = (sd / len(arg)).tolist()
    r.error = (np.sqrt(se) / sd)
    return r





