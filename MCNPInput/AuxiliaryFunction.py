import numpy as np
import re

def getXTran(theta:float):
    """
    获取旋转矩阵
    :param theta: 以x轴为旋转中心顺时针旋转的角度
    :return: 三维空间的旋转矩阵
    """
    rad = theta / 180 * np.pi
    _cos = np.cos(rad)
    _sin = np.sin(rad)
    return np.array([[1, 0, 0],
                     [0, _cos, -_sin],
                     [0, _sin, _cos]])

def getYTran(theta:float):
    """
    获取旋转矩阵
    :param theta: 以y轴为旋转中心顺时针旋转的角度
    :return: 三维空间的旋转矩阵
    """
    rad = theta / 180 * np.pi
    _cos = np.cos(rad)
    _sin = np.sin(rad)
    return np.array([[_cos, 0, _sin],
                     [0, 1, 0],
                     [-_sin, 0, _cos]])

def getZTran(theta:float):
    """
    获取旋转矩阵
    :param theta: 以z轴为旋转中心顺时针旋转的角度
    :return: 三维空间的旋转矩阵
    """
    rad = theta / 180 * np.pi
    _cos = np.cos(rad)
    _sin = np.sin(rad)
    return np.array([[_cos, -_sin, 0],
                     [_sin, _cos, 0],
                     [0, 0, 1]])

def transformationMatrix(x_angle:float = 0, y_angle:float = 0, z_angle:float = 0):
    xy = np.dot(getXTran(x_angle), getYTran(y_angle))
    xyz = np.dot(xy, getZTran(z_angle))
    return xyz

def replace_char(old_string, char, index):
    '''
    字符串按索引位置替换字符
    :param old_string: 输入的字符串
    :param char: 替换字符
    :param index: 被替换字符索引
    '''
    return f'{old_string[:index]}{char}{old_string[index+1:]}'

def line_split(ss:str, limit:int = 80, f:int = 15):
    """
     对于过长的一行字符串切分成在MCNP中多行换行的字符串
    :param ss: 一行字符串
    :param limit: 每行最大字符数量
    :param f: 检查最后f个字符数，从一个连续的单词后进行换行
    """
    rs = ""
    lines = ss.split('\n')
    output = []
    for line in lines:
        if len(line) == 0:
            continue
        rs = ''
        if len(line) > 0 and line[0].lower() != 'c':
            line_dataIdx = line.find("$")
            if line_dataIdx != -1:
                line_note = line[line_dataIdx:]
                line = line[:line_dataIdx]
            else:
                line_note = None
            if len(line) >= limit:
                line_param = line.split()
                startIdx = 0
                for i in range(len(line_param)):
                    _tmpLine = " ".join(line_param[startIdx:i])
                    if len(_tmpLine) >= limit:
                        rs += " ".join(line_param[startIdx:i-1]) + "&\n"
                        startIdx = i-1
                rs += " ".join(line_param[startIdx:])
            else:
                rs = line
            if line_note is not None:
                rs += line_note
        else:
            rs = line
        output.append(rs)
    return '\n'.join(output)

# 将分行的数据对接上，同时去掉注释
def line_merge(ss: str):
    lines = ss.split('\n')
    before = re.compile(r'^\s+.*')
    lines_after = ''
    flag = False
    for i in range(len(lines)):
        line = lines[i]
        if len(line) > 0 and line[0].lower() == 'c':
            continue
        else:
            endIdx = line.find(r'$')
            if endIdx != -1:
                line = line[:endIdx]
            if len(line) > 0 and line[-1] =='&':
                flag = True
            if before.match(line) or flag:
                lines_after += line
                flag = False
            else:
                lines_after += '\n{}'.format(line)
    if lines_after[0] == '\n':
        lines_after = lines_after[1:]
    return lines_after