import numpy as np
import pandas as pd

from MCNPInput import GeometricModel, AuxiliaryFunction, SourceDefine
from utility import readSCC, stratumCellPackage, outputReader, stratumModel, fastForwardModeling
import os, subprocess
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import copy
import sys
import shutil

if sys.platform == 'win32':
    mcnp_order = "mcnp6.exe"
    mcnp_mpi_order = "mcnp6_mpi"
elif sys.platform == 'linux':
    mcnp_order = "mcnp6"
    mcnp_mpi_order = "mcnp6.mpi"

class AMS_Manager(object):
    def __init__(self, model:stratumModel.StratumModel, name:str = "tmpAMS"):
        '''
        管理运算进程的类
        :param model: 描述地层模型以及井眼轨迹的类
        :param name: 本次计算的名称，会根据名称创建对应的文件夹
        '''
        self.name = name
        if not os.path.isdir("./{}".format(self.name)):
            os.makedirs("./{}".format(self.name))
        self.stratumModel = model # readSCC.LayerModel(file, track, radius_of_investigation, well_diameter)
        # 标准结果，即探测范围只有一个地层时的情况
        self.standResult_data = None
        self.standResult_error = None
        # 截至粒子数
        self.nps = np.power(2, 22)
        # 快速计算模型权重
        self.ffms:list[fastForwardModeling.fastForwardModeling] = []
        # mpi hostfile
        self.mpiFile = None

    def set_mpi(self, hostFile:str = None):
        '''
        设置mpihostfile， 设置后会使用mpi来运行mcnp
        :param hostFile:
        :return:
        '''
        if os.path.exists(hostFile):
            self.mpiFile = hostFile
        else:
            self.mpiFile = None

    def run_mcnp(self, justOutput:bool = False, singleFormationAccelration:bool=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        在本地运行mcnp
        :param justOutput:只输出mcnp输入文件，不计算
        :param singleFormationAccelration: 是否对只有单个地层的探测点加速
        :return:
        '''
        data = []
        error = []
        index = []
        runRootPath = "./{}".format(self.name)
        # 遍历每一个探测点
        _l = len(self.stratumModel)
        for i in tqdm(range(_l), postfix=self.name):
            model = self.stratumModel[i] # 对应探测点的模型
            _depth = self.stratumModel.trajectory.track.index[i] # 对应探测点的深度坐标数据
            _flag_saveStand = False # 标志：是否有单地层模板
            _flag_calculate = True # 标志：是否需要使用mcnp计算
            _data_a = [] # 临时存储数据
            _error_a = [] # 临时存储误差

            # 如果当前测量点为单地层并且开启了单地层加速
            if len(model.stratumBorders) == 1 and singleFormationAccelration:
                if self.standResult_data is not None:
                    _data_a = copy.deepcopy(self.standResult_data)
                    _error_a = copy.deepcopy(self.standResult_error)
                    _flag_calculate = False
                else:
                    _flag_saveStand = True

            # 需要使用mcnp计算
            if _flag_calculate:
                # 对于八方位gamma情况，一个模型只有四个探测器，需要两个模型计算分别计算
                for angel in [0, 45]:
                    # mcnp的模型构建，这里将模型分三部分建立
                    ## 第一部分为地层部分
                    cellP_Stratum = stratumCellPackage.StratumInput(model)
                    ## 第二部分为井眼部分
                    cellP_borehole = stratumCellPackage.Borehole(model.zone.innerDiameter, model.zone.Radius, parent=cellP_Stratum)
                    ## 第三部分为探测器部分，两个模型探测器旋转的角度不同，从而测量八方位gamma
                    detectorTrans = GeometricModel.MCNP_transformation(AuxiliaryFunction.transformationMatrix(y_angle=angel), -1)
                    cellP_detector = stratumCellPackage.BaseDetector(parent=cellP_borehole,transform=detectorTrans)

                    # 获取cardsPool
                    mcnp_cards = cellP_Stratum.get_cardsPool()
                    # 基于当前设置更改模拟粒子数量
                    mcnp_cards.TallyCard.cut_nps = self.nps

                    mcnp_input_filename = "tmpMCNPInput_{}_{:d}_a{}.txt".format(self.name, i, angel)
                    mcnp_output_filename = "tmpMCNPOutput_{}_{:d}_a{}.txt".format(self.name, i, angel)
                    runtp_filename = "runtpe_{}_{:d}_a{}".format(self.name, i, angel) # 中间文件
                    out_filename = "outp" # 中间文件

                    # 如果不是只需要生成mcnp输入文件，那么需要删除中间文件
                    if not justOutput:
                        for _name in [runtp_filename, out_filename]:
                            _path = os.path.join(runRootPath, _name)
                            if os.path.exists(_path):
                                os.remove(_path)
                    # 存储输入文件
                    mcnp_str = str(mcnp_cards)  # mcnp输入文件内容
                    with open(os.path.join(runRootPath, mcnp_input_filename), 'w') as file:
                        file.write(mcnp_str)
                    # 如果不是只需要生成mcnp输入文件并且对应的输出文件不存在，则调用mcnp计算
                    if not justOutput and not os.path.exists(os.path.join(runtp_filename, mcnp_output_filename)):
                        if self.mpiFile is None:
                            order = "cd {rootPath}; {exeOrder} inp={inputFile} outp={outputFile} run={runFile} tasks {cpuCores:d}".format(
                                rootPath = runRootPath, exeOrder = mcnp_order, inputFile = mcnp_input_filename, outputFile = mcnp_output_filename,
                                runFile = runtp_filename, cpuCores = cpu_count())
                        else:
                            order = "cd {rootPath}; mpirun -f {hostFile} {exeOrder} i={inputFile} outp={outputFile} run={runFile}".format(
                                rootPath=runRootPath, inputFile=mcnp_input_filename,exeOrder="mcnp6.mpi",
                                outputFile=mcnp_output_filename, hostFile=os.path.abspath(self.mpiFile),runFile = runtp_filename)
                        sr = subprocess.run(order, shell=True, stdout=subprocess.PIPE)
                        if sr.returncode != 0:
                            os.remove(os.path.join(runRootPath,mcnp_output_filename))
                            sr = subprocess.run(order, shell=True, stdout=subprocess.PIPE)
                        # mcnp计算程序正常结束，则读取输出文件
                        if sr.returncode == 0:
                            reader = outputReader.MCNP_OutputReader(os.path.join(runRootPath,mcnp_output_filename))
                            _data_a.extend(reader.data)
                            _error_a.extend(reader.error)
                        # 删除中间文件
                        if not justOutput:
                            for _name in [runtp_filename, out_filename]:
                                _path = os.path.join(runRootPath, _name)
                                if os.path.exists(_path):
                                    os.remove(_path)
                    # 如果是只需要生成mcnp输入文件但是对应的输出文件已经存在，则读取输出文件
                    elif os.path.exists(os.path.join(runRootPath,mcnp_output_filename)):
                        reader = outputReader.MCNP_OutputReader(os.path.join(runRootPath,mcnp_output_filename))
                        _data_a.extend(reader.data)
                        _error_a.extend(reader.error)
                # 对于第一次计算单地层数据时，将单地层数据保存
                if _flag_saveStand:
                    self.standResult_error = _error_a
                    self.standResult_data = _data_a
            # 将计算结果根据模型总放射水平调整数值
            thisData = list(map(lambda x: x * model.get_total_doseStand(), _data_a))
            # 将测量点数据整合到总数据对象中
            data.append(thisData)
            error.append(_error_a)
            index.append(_depth)

        d = pd.DataFrame(data, index=index)
        e = pd.DataFrame(error, index=index)
        return d, e

    def create_WWOUT(self, circulation:int = 1,justOutput:bool = False):
        '''
        创建对应的快速正演模板数据
        :param circulation: 快速正演权重模板的迭代次数
        :param justOutput: 是否只输出输入文件
        :return:
        '''
        # 快速正演为单地层模型，只需要通过探测空间对象建立地层
        zone = self.stratumModel[0].zone
        self.ffms.clear()
        print("wwout start")
        os.mkdir("./{}/wwg".format(self.name))
        wwgRootPath = "./{}/wwg".format(self.name)

        angels = [0, 45]
        # 八方位探测器需要两个探测器模型
        for angel in angels:
            # 第一部分，wwg地层，可以分出单独探测器的input
            cellP_Stratum = stratumCellPackage.WWGCreatorInput(zone)
            # 第二部分，井眼
            cellP_borehole = stratumCellPackage.Borehole(zone.innerDiameter, zone.Radius,
                                                         parent=cellP_Stratum)
            # 第三部分，探测器
            detectorTrans = GeometricModel.MCNP_transformation(AuxiliaryFunction.transformationMatrix(y_angle=angel), -1)
            cellP_detector = stratumCellPackage.BaseDetector(parent=cellP_borehole, transform=detectorTrans)

            # 第四部分， 划分网格
            wwgCard = SourceDefine.MCNP_wwgCard(cellP_Stratum.tally)
            wwgCard.meshArg["GEOM"] = "CYL"
            wwgCard.meshArg["REF"] = "0 0 0"
            wwgCard.meshArg["ORIGIN"] = "0 {} 0".format(zone.downSurface)
            wwgCard.meshArg["AXS"] = "0 1 0"
            wwgCard.meshArg["VEC"] = "1 0 0"

            wwgCard.splitArg = ["{} {}".format(zone.innerDiameter,zone.Radius), "{:d} {:d}".format(1, int(zone.Radius / 5)),  # 径向
                                "{}".format(zone.topSurface - zone.downSurface), "{:d}".format(int( (zone.topSurface - zone.downSurface) / 5 )), # 轴向
                                "{}".format(1), "{:d}".format(16)] # 角度
            cellP_Stratum.tally.tail.append(wwgCard)
            cellP_Stratum.tally.cut_nps = self.nps
            # 获取一系列cardsPool，对应每一个探测器
            _cardPools = cellP_Stratum.get_wwgCardPool()



            if circulation > 0:
                pbar = tqdm(total=len(_cardPools) * circulation)
                pbar.set_postfix_str("angel: {} in [{}]".format(angel, ",".join(list(map(str, angels)))))

            # 对每一个探测器的cardsPool
            for i in range(len(_cardPools)):
                # 输入文件路径
                mcnp_str = str(_cardPools[i])
                input_filename = "tmpMCNPwwgInput_{}_{:d}_a{}.txt".format(self.name, i, angel)
                mcnp_input_path = os.path.join(wwgRootPath, input_filename)#"./{}/wwg/{}".format(self.name, input_filename)
                # 建立mcnp输入文件
                if not os.path.exists(mcnp_input_path):
                    with open(mcnp_input_path, 'w') as file:
                        file.write(mcnp_str)
                # 迭代权重数据
                j = 0 # 迭代次数记录
                while True:
                    # 从第二次开始，需要加入权重文件，输入文件加入一行参数
                    if j == 1:
                        with open(mcnp_input_path, 'a+') as file:
                            file.write("\nwwp:p  5 3 5 0 -1")
                    # 从第二次开始，需要读取上一次的输出作为参考
                    if j >= 1:
                        past_wwout_path = wwout_path
                        past_wwout_filename = wwout_filename

                    # 建立输出文件路径
                    output_filename = "tmpMCNPwwgOutput_{}_{:d}_a{}_c{:d}.txt".format(self.name, i, angel, j)
                    wwout_filename = "wwout_{}_{:d}_a{}_c{:d}.txt".format(self.name, i , angel, j)
                    mcnp_output_path = os.path.join(wwgRootPath, output_filename)#"./{}/wwg/{}".format(self.name, output_filename)
                    wwout_path = os.path.join(wwgRootPath, wwout_filename)#"./{}/wwg/{}".format(self.name, wwout_filename)
                    # 中间文件路径
                    runtp_filename = "runtpe_{}_{}".format(i, j)
                    runtp_path = os.path.join(wwgRootPath, runtp_filename)#"./wwg/{}".format(runtp_filename)
                    # 删除临时文件
                    for _path in [runtp_path, mcnp_output_path]:
                        if os.path.exists(_path):
                            os.remove(_path)
                    # 开始计算
                    if not justOutput and not os.path.exists(wwout_path):
                        if j == 0:
                            wwinp = ""
                        else:
                            wwinp = "wwinp={}".format(past_wwout_filename)
                        # 正常计算
                        if self.mpiFile is None:
                            order = "cd {rootPath}; {order} inp={inputPath} outp={outputPath} run=runtpe wwout={wwoutPath} {wwinp} tasks {tasks:d}".format(
                                rootPath=wwgRootPath,
                                order=mcnp_order,
                                inputPath=input_filename,
                                outputPath=output_filename,
                                runFile=runtp_filename,
                                wwoutPath=wwout_filename,
                                wwinp=wwinp,
                                tasks=cpu_count())
                        else:
                            order = "cd {rootPath}; mpirun -f {hostFile} {order} inp={inputPath} outp={outputPath} run={runFile} wwout={wwoutPath} {wwinp}".format(
                                rootPath=wwgRootPath,
                                hostFile=os.path.abspath(self.mpiFile),
                                order=mcnp_mpi_order,
                                inputPath=input_filename,
                                outputPath=output_filename,
                                runFile=runtp_filename,
                                wwinp=wwinp,
                                wwoutPath=wwout_filename)
                        sr = subprocess.run(order, shell=True, stdout=subprocess.PIPE)
                    # 删除临时文件
                    if os.path.exists(runtp_path):
                        os.remove(runtp_path)

                    # 退出循环条件
                    j += 1

                    if circulation > 0:
                        pbar.update(1)
                        if j >= circulation:
                            break
                    else:

                        tmp_out = outputReader.MCNP_WWOUTReader(wwout_path)
                        tmp_s = np.sum(tmp_out.data == 0.)
                        l_ = 1
                        for i in tmp_out.data.shape:
                            l_ += l_ * i

                        _a_str = "angel: {} in [{}]".format(angel, ",".join(list(map(str, angels))))
                        print("\r{} wwout check: {:.2%}, Exit condition: < {:.2%}".format(_a_str, tmp_s / float(l_), 0.05))

                        if (tmp_s / float(l_)) < 0.05:
                            break
                if not justOutput:
                    # 转存数据
                    dst_path = os.path.join(wwgRootPath, "wwout_{:d}_a{}.txt".format(i, angel))
                    shutil.copy(wwout_path, dst_path)
                    dst_path = os.path.join(wwgRootPath, "ouput_{:d}_a{}.txt".format(i, angel))
                    shutil.copy(os.path.abspath(mcnp_output_path), dst_path)
        if not justOutput:
            # 载入最终的数据
            self.load_wwout(wwgRootPath)

    def load_wwout(self,path:str = None, detector:int=4):
        self.ffms.clear()
        if path is None:
            path = "./{}/wwg".format(self.name)
        for angel in [0, 45]:
            for i in range(detector):
                wwout_path = os.path.join(path, "wwout_{:d}_a{}.txt".format(i, angel))
                rwwout = outputReader.MCNP_WWOUTReader(wwout_path)
                mcnp_output_path = os.path.join(path,"ouput_{:d}_a{}.txt".format(i, angel))
                rout = outputReader.MCNP_OutputReader(mcnp_output_path)
                zone = self.stratumModel[0].zone
                ffm = fastForwardModeling.fastForwardModeling(rwwout, rout, zone)
                self.ffms.append(ffm)

    def run_fast(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = []
        error = []
        index = []
        # 遍历每一个探测点
        _l = len(self.stratumModel)
        with Pool(cpu_count()) as p:
            r = p.map(self._ffm_data, range(_l))
        for i in range(_l):
            _tmpData, _tmpError = r[i]#self._ffm_data(i)
            depth = self.stratumModel.trajectory.track.index[i]  # 对应探测点的深度坐标数据
            data.append(_tmpData)
            error.append(_tmpError)
            index.append(depth)
        d_pd = pd.DataFrame(data, index=index)
        e_pd = pd.DataFrame(error, index=index)
        return d_pd, e_pd

    @staticmethod
    def _run_complete(layer:pd.DataFrame, api:np.array,track:stratumModel.WellTrack, boxRange:np.array, name:str,
                     justOutput:bool=False,nps:int=np.power(2,26), mpiFile:str=None, stratumAzi:np.array = None):
        data = []
        error = []
        index = []
        runRootPath = "./{}".format(name)
        if not os.path.isdir(runRootPath):
            os.makedirs(runRootPath)

        for i in tqdm(range(len(track.track)), postfix=name):
            _data_a = []  # 临时存储数据
            _error_a = []  # 临时存储误差
            _depth = track.track.index[i]  # 对应探测点的深度坐标数据

            for angel in [0, 45]:
                mcnp_input_filename = "tmpMCNPInput_{}_{:d}_a{}.txt".format(name, i, angel)
                mcnp_output_filename = "tmpMCNPOutput_{}_{:d}_a{}.txt".format(name, i, angel)
                runtp_filename = "runtpe_{}_{:d}_a{}".format(name, i, angel)  # 中间文件
                out_filename = "outp"  # 中间文件

                x = track.track["X"].iloc[i]
                y = track.track["Y"].iloc[i]
                z = track.track["Z"].iloc[i]

                model = stratumCellPackage.StratumCompleteInput(layer, api, boxRange, stratumAzi=stratumAzi)
                bh = stratumCellPackage.Borehole_complete(boxRange, model)
                detectorTrans = GeometricModel.MCNP_transformation(
                    AuxiliaryFunction.transformationMatrix(y_angle=angel), x - 1, y, z)
                cellP_detector = stratumCellPackage.BaseDetector(parent=bh, transform=detectorTrans)

                pool = model.get_cardsPool()
                pool.nps = nps
                mcnp_str = str(pool)
                with open(os.path.join(runRootPath, mcnp_input_filename), 'w') as file:
                    file.write(mcnp_str)

                if mpiFile is None:
                    order = "cd {rootPath}; {exeOrder} inp={inputFile} outp={outputFile} run={runFile} tasks {cpuCores:d}".format(
                        rootPath=runRootPath, exeOrder=mcnp_order, inputFile=mcnp_input_filename,
                        outputFile=mcnp_output_filename,
                        runFile=runtp_filename, cpuCores=cpu_count())
                else:
                    order = "cd {rootPath}; mpirun -f {hostFile} {exeOrder} i={inputFile} outp={outputFile} run={runFile}".format(
                        rootPath=runRootPath, inputFile=mcnp_input_filename,exeOrder="mcnp6.mpi",
                        outputFile=mcnp_output_filename, hostFile=os.path.abspath(mpiFile), runFile=runtp_filename)

                if not justOutput:
                    sr = subprocess.run(order, shell=True, stdout=subprocess.PIPE)
                    # mcnp计算程序正常结束，则读取输出文件
                    if sr.returncode == 0:
                        reader = outputReader.MCNP_OutputReader(os.path.join(runRootPath, mcnp_output_filename))
                        _data_a.extend(reader.data)
                        _error_a.extend(reader.error)
            # 将测量点数据整合到总数据对象中
            data.append(_data_a)
            error.append(_error_a)
            index.append(_depth)
        d = pd.DataFrame(data, index=index)
        e = pd.DataFrame(error, index=index)
        return d, e

    @staticmethod
    def run_complete(layer:pd.DataFrame, api:np.array,track:stratumModel.WellTrack, boxRange:np.array, name:str,
                     justOutput:bool=False,nps:int=np.power(2,26), mpiFile:str=None, stratumAzi:np.array = None):
        model = stratumModel.StratumModel_DataFrame_complete(layer, api, track, boxRange[3], boxRange[2], topSurface=boxRange[4], downSurface=boxRange[5], stratumAzi=stratumAzi)
        ssModel = model[0]

        data = []
        error = []
        index = []
        runRootPath = "./{}".format(name)
        if not os.path.isdir(runRootPath):
            os.makedirs(runRootPath)

        for i in tqdm(range(len(track.track)), postfix=name):
            _data_a = []  # 临时存储数据
            _error_a = []  # 临时存储误差
            _depth = track.track.index[i]  # 对应探测点的深度坐标数据

            for angel in [0, 45]:
                mcnp_input_filename = "tmpMCNPInput_{}_{:d}_a{}.txt".format(name, i, angel)
                mcnp_output_filename = "tmpMCNPOutput_{}_{:d}_a{}.txt".format(name, i, angel)
                runtp_filename = "runtpe_{}_{:d}_a{}".format(name, i, angel)  # 中间文件
                out_filename = "outp"  # 中间文件

                x = track.track["X"].iloc[i]
                y = track.track["Y"].iloc[i]
                z = track.track["Z"].iloc[i]

                ## 第一部分为地层部分
                cellP_Stratum = stratumCellPackage.StratumInput(ssModel)
                bh = stratumCellPackage.Borehole_complete(boxRange, cellP_Stratum)
                detectorTrans = GeometricModel.MCNP_transformation(
                    AuxiliaryFunction.transformationMatrix(y_angle=angel), x - 1, y, z)
                cellP_detector = stratumCellPackage.BaseDetector(parent=bh, transform=detectorTrans)

                pool = cellP_Stratum.get_cardsPool()
                pool.nps = nps
                mcnp_str = str(pool)
                with open(os.path.join(runRootPath, mcnp_input_filename), 'w') as file:
                    file.write(mcnp_str)

                if mpiFile is None:
                    order = "cd {rootPath}; {exeOrder} inp={inputFile} outp={outputFile} run={runFile} tasks {cpuCores:d}".format(
                        rootPath=runRootPath, exeOrder=mcnp_order, inputFile=mcnp_input_filename,
                        outputFile=mcnp_output_filename,
                        runFile=runtp_filename, cpuCores=cpu_count())
                else:
                    order = "cd {rootPath}; mpirun -f {hostFile} {exeOrder} i={inputFile} outp={outputFile} run={runFile}".format(
                        rootPath=runRootPath, inputFile=mcnp_input_filename, exeOrder="mcnp6.mpi",
                        outputFile=mcnp_output_filename, hostFile=os.path.abspath(mpiFile), runFile=runtp_filename)

                if not justOutput:
                    sr = subprocess.run(order, shell=True, stdout=subprocess.PIPE)
                    # mcnp计算程序正常结束，则读取输出文件
                    if sr.returncode == 0:
                        reader = outputReader.MCNP_OutputReader(os.path.join(runRootPath, mcnp_output_filename))
                        _data_a.extend(reader.data)
                        _error_a.extend(reader.error)
                # 将测量点数据整合到总数据对象中

            data.append(_data_a)
            error.append(_error_a)
            index.append(_depth)

        d = pd.DataFrame(data, index=index)
        e = pd.DataFrame(error, index=index)
        return d, e

    def test_fast_InIt(self):
        runRootPath = "./{}".format(self.name)
        if len(self.ffms) == 0:
            self.create_WWOUT(5)
        _f0 = self.ffms[0]
        # 遍历每一个探测点
        _l = len(self.stratumModel)
        for i in tqdm(range(_l), postfix=self.name):
            model = self.stratumModel[i]  # 对应探测点的模型
            _depth = self.stratumModel.trajectory.track.index[i]  # 对应探测点的深度坐标数据
            Dir_nowDepth = os.path.join(runRootPath, "index_{}_{}".format(_depth,i))
            os.mkdir(Dir_nowDepth)
            # 对于八方位gamma情况，一个模型只有四个探测器，需要两个模型计算分别计算
            for angel in [0, 45]:
                # mcnp的模型构建，这里将模型分三部分建立
                ## 第一部分为地层部分
                cellP_Stratum = stratumCellPackage.StratumInput(model)
                ## 第二部分为井眼部分
                cellP_borehole = stratumCellPackage.Borehole(model.zone.innerDiameter, model.zone.Radius,
                                                             parent=cellP_Stratum)
                ## 第三部分为探测器部分，两个模型探测器旋转的角度不同，从而测量八方位gamma
                detectorTrans = GeometricModel.MCNP_transformation(
                    AuxiliaryFunction.transformationMatrix(y_angle=angel), -1)
                cellP_detector = stratumCellPackage.BaseDetector(parent=cellP_borehole, transform=detectorTrans)

                # 获取cardsPool
                mcnp_cards = cellP_Stratum.get_cardsPool()
                # 基于当前设置更改模拟粒子数量
                mcnp_cards.TallyCard.cut_nps = self.nps

                Dir_angel = os.path.join(Dir_nowDepth, "angel_{}".format(angel))
                os.mkdir(Dir_angel)




                for layerIndex in range(len(model)):
                    layer = model[layerIndex]
                    Dir_layer = os.path.join(Dir_angel, "layer_{}".format(layerIndex))
                    os.mkdir(Dir_layer)

                    mcnp_input_filename = "tmpMCNPInput.txt"
                    mcnp_input_path = os.path.join(Dir_layer, mcnp_input_filename)
                    # 存储输入文件
                    mcnp_str = str(mcnp_cards)  # mcnp输入文件内容
                    with open(mcnp_input_path, 'w') as file:
                        file.write(mcnp_str)
                        file.write("\nwwp:p  5 3 5 0 -1")

                    cp_weight = copy.deepcopy(_f0.weight)
                    result = _f0.getInIt(layer)
                    cp_weight.data = result
                    wwinp_path = os.path.join(Dir_layer, "wwinp")
                    cp_weight.save(wwinp_path)

    def test_api(self):
        data = []
        error = []
        index = []
        # 遍历每一个探测点
        _l = len(self.stratumModel)
        for i in tqdm(range(_l), postfix=self.name):
            model = self.stratumModel[i]  # 对应探测点的模型
            depth = self.stratumModel.trajectory.track.index[i]  # 对应探测点的深度坐标数据
            if len(self.ffms) == 0:
                self.create_WWOUT(5)
            _tmpData = np.empty((len(self.ffms),))
            _tmpError = np.empty((len(self.ffms),))
            for layer in model:
                _tmpData_layer = []
                _tmpError_layer = []
                for ffm in self.ffms:
                    _d = ffm.data
                    d = layer.api
                    e = ffm.stand.error[0]
                    _tmpData_layer.append(d)
                    _tmpError_layer.append(np.power(e * d, 2))
                _tmpData = np.array(_tmpData_layer)
                _tmpError += np.array(_tmpError_layer)
            _tmpError = np.sqrt(_tmpError) / _tmpData
            data.append(_tmpData)
            error.append(_tmpError)
            index.append(depth)
        d_pd = pd.DataFrame(data, index=index)
        e_pd = pd.DataFrame(error, index=index)
        return d_pd, e_pd

    def test_weight(self):
        data = []
        error = []
        index = []
        # 遍历每一个探测点
        _l = len(self.stratumModel)
        for i in tqdm(range(_l), postfix=self.name):
            model = self.stratumModel[i]  # 对应探测点的模型
            depth = self.stratumModel.trajectory.track.index[i]  # 对应探测点的深度坐标数据
            if len(self.ffms) == 0:
                self.create_WWOUT(5)
            _tmpData = np.empty((len(self.ffms),))
            _tmpError = np.empty((len(self.ffms),))
            for layer in model:
                _tmpData_layer = []
                _tmpError_layer = []
                checkVix = self.ffms[0].getInIt(layer)
                for ffm in self.ffms:
                    _d = ffm.data
                    d = np.sum(_d * checkVix)
                    e = ffm.stand.error[0]
                    _tmpData_layer.append(d)
                    _tmpError_layer.append(np.power(e * d, 2))
                _tmpData = np.array(_tmpData_layer)
                _tmpError = np.array(_tmpError_layer)
            _tmpError = np.sqrt(_tmpError) / _tmpData
            data.append(_tmpData)
            error.append(_tmpError)
            index.append(depth)
        d_pd = pd.DataFrame(data, index=index)
        e_pd = pd.DataFrame(error, index=index)
        return d_pd, e_pd

    def _ffm_data(self, i:int) -> tuple[np.array, np.array]:
        if i >= 42:
            _tm = 0
        model = self.stratumModel[i]  # 对应探测点的模型
        depth = self.stratumModel.trajectory.track.index[i]  # 对应探测点的深度坐标数据
        if len(self.ffms) == 0:
            self.create_WWOUT(5)
        _tmpData = np.zeros((len(self.ffms),))
        _tmpError = np.zeros((len(self.ffms),))
        for layer in model:
            _tmpData_layer = []
            _tmpError_layer = []
            checkVix = self.ffms[0].getInIt(layer)
            for ffm in self.ffms:
                _d = ffm.data
                d = np.sum(_d * checkVix) * ffm.normalCoe * layer.api
                e = ffm.stand.error[0]
                _tmpData_layer.append(d)
                _tmpError_layer.append(np.power(e * d, 2))
            _tmpData += np.array(_tmpData_layer)
            _tmpError += np.array(_tmpError_layer)
        _tmpError = np.sqrt(_tmpError) / _tmpData
        return _tmpData, _tmpError



class AMS_CIFLog_Manager(AMS_Manager):
    def __init__(self,file:str, trajectoryDir:str, radius_of_investigation:float = 80, well_diameter:float = 10):
        hPath = os.path.join(trajectoryDir, 'HDRIFT.txt')
        vPath = os.path.join(trajectoryDir, 'VDRIFT.txt')
        track = readSCC.CIFLogTrack(hPath,vPath)
        _stratumModel = readSCC.LayerModel(file, track, radius_of_investigation, well_diameter)
        super().__init__(self, _stratumModel)
