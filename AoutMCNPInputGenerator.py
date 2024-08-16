import MCNPInput.MCNPAutoInput
import utility
import MCNPInput.MCNPAutoInput
import MCNPInput.GeometricModel
import MCNPInput.Material
from abc import abstractmethod, ABCMeta


class InputGenerator(metaclass=ABCMeta):
    def __init__(self, stratum_mode:utility.readSCC.smallModel):
        self.stratum_mode = stratum_mode


    @abstractmethod
    def base_model(self) -> MCNPInput.MCNPAutoInput.MCNP_CardsPool:
        # 1.创建探测器
        # 2.设置粒子源
        # 3.地层默认材质
        pass

class WellLoggingGammaInputGenerator(InputGenerator):
    def base_model(self) -> MCNPInput.MCNPAutoInput.MCNP_CardsPool:
        b = MCNPInput.MCNPAutoInput.MCNP_CardsPool()
        # 钻井液导流通道
        diversionChannel = MCNPInput.GeometricModel.MCNP_cell(
            MCNPInput.Material.create_material_from_string("water"),
            1.
        )
        diversionChannel.addSurface(
            MCNPInput.GeometricModel.MCNP_PlaneSurface('pz', ['-40'])
        )
        diversionChannel.addUnionSurface()